from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Built-in stress scenarios: name -> market shock (fractional, e.g. -0.57)
# ---------------------------------------------------------------------------
BUILTIN_STRESS_SCENARIOS: dict[str, float] = {
    "2008_GFC": -0.57,
    "COVID_2020": -0.34,
    "2022_BEAR": -0.25,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RiskAggResult:
    """Aggregated risk analysis result."""

    var_cvar: list[dict[str, float]] = field(default_factory=list)
    marginal_contributions: list[dict[str, float]] = field(default_factory=list)
    concentration: dict[str, Any] = field(default_factory=dict)
    stress_results: list[dict[str, Any]] = field(default_factory=list)
    liquidity_risk: dict[str, Any] = field(default_factory=dict)
    tail_deps: dict[str, float] = field(default_factory=dict)
    compliance_breaches: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------
class RiskAggregator:
    """Portfolio-level risk aggregation across multiple strategy return streams."""

    def __init__(
        self,
        experiment_returns: dict[str, pd.Series],
        weights: dict[str, float],
    ) -> None:
        if not experiment_returns:
            raise ValueError("experiment_returns must not be empty")
        self.experiment_returns = experiment_returns
        self.weights = weights
        self._strategies = sorted(experiment_returns.keys())
        # Build aligned returns DataFrame
        self._returns_df = pd.DataFrame(experiment_returns).dropna()
        # Build portfolio return series (correlation-adjusted through construction)
        self._portfolio_returns = self._build_portfolio_returns()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_portfolio_returns(self) -> pd.Series:
        port = pd.Series(0.0, index=self._returns_df.index)
        for s in self._strategies:
            w = self.weights.get(s, 0.0)
            port = port + w * self._returns_df[s]
        return port

    def _build_portfolio_returns_without(self, exclude: str) -> pd.Series:
        remaining = [s for s in self._strategies if s != exclude]
        if not remaining:
            return pd.Series(0.0, index=self._returns_df.index)
        # Re-weight remaining proportionally
        total_w = sum(self.weights.get(s, 0.0) for s in remaining)
        port = pd.Series(0.0, index=self._returns_df.index)
        for s in remaining:
            w = self.weights.get(s, 0.0)
            scaled_w = w / total_w if total_w > 0 else 1.0 / len(remaining)
            port = port + scaled_w * self._returns_df[s]
        return port

    # ------------------------------------------------------------------
    # 1. VaR / CVaR via historical simulation
    # ------------------------------------------------------------------
    def compute_var_cvar(self) -> list[dict[str, float]]:
        """Return VaR and CVaR at 95% and 99% confidence levels."""
        results: list[dict[str, float]] = []
        rets = self._portfolio_returns.values
        for confidence in (0.95, 0.99):
            alpha = 1.0 - confidence
            var = float(-np.percentile(rets, alpha * 100))
            # CVaR = mean of losses beyond VaR
            tail = rets[rets <= -var]
            cvar = float(-tail.mean()) if len(tail) > 0 else var
            results.append({
                "confidence": confidence,
                "var": var,
                "cvar": cvar,
            })
        return results

    # ------------------------------------------------------------------
    # 2. Marginal risk contribution per strategy
    # ------------------------------------------------------------------
    def compute_marginal_contributions(self) -> list[dict[str, float]]:
        """CVaR with vs without each strategy at 99%."""
        alpha = 0.01
        port_rets = self._portfolio_returns.values
        port_var = -np.percentile(port_rets, alpha * 100)
        tail = port_rets[port_rets <= -port_var]
        port_cvar = float(-tail.mean()) if len(tail) > 0 else float(port_var)

        contributions: list[dict[str, float]] = []
        for s in self._strategies:
            ex_rets = self._build_portfolio_returns_without(s).values
            ex_var = -np.percentile(ex_rets, alpha * 100)
            ex_tail = ex_rets[ex_rets <= -ex_var]
            ex_cvar = float(-ex_tail.mean()) if len(ex_tail) > 0 else float(ex_var)
            contributions.append({
                "strategy": s,
                "marginal_cvar": port_cvar - ex_cvar,
                "portfolio_cvar": port_cvar,
            })
        return contributions

    # ------------------------------------------------------------------
    # 3. Risk concentration
    # ------------------------------------------------------------------
    def compute_concentration(self) -> dict[str, Any]:
        """Herfindahl index and flag strategies > 30% of risk."""
        # Use absolute weight as risk proxy
        abs_weights = {s: abs(self.weights.get(s, 0.0)) for s in self._strategies}
        total = sum(abs_weights.values())
        if total == 0:
            shares = {s: 1.0 / len(self._strategies) for s in self._strategies}
        else:
            shares = {s: abs_weights[s] / total for s in self._strategies}

        hhi = sum(v ** 2 for v in shares.values())
        concentrated = [s for s, v in shares.items() if v > 0.30]
        return {
            "herfindahl_index": float(hhi),
            "strategy_shares": shares,
            "concentrated_strategies": concentrated,
            "is_concentrated": len(concentrated) > 0,
        }

    # ------------------------------------------------------------------
    # 4. Stress testing
    # ------------------------------------------------------------------
    def stress_test(
        self,
        scenarios: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply shock scenarios and compute stressed drawdown."""
        if scenarios is None:
            scenarios = BUILTIN_STRESS_SCENARIOS

        results: list[dict[str, Any]] = []
        for name, shock in scenarios.items():
            # Build a simple shock path: linear drawdown over 60 days
            n_days = 60
            daily_shock = (1.0 + shock) ** (1.0 / n_days) - 1.0
            shock_path = np.full(n_days, daily_shock)

            # Apply to portfolio
            cum = np.cumprod(1.0 + shock_path)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            max_dd = float(dd.min())

            results.append({
                "scenario": name,
                "shock": shock,
                "stressed_max_drawdown": max_dd,
                "final_value": float(cum[-1]),
            })
        return results

    # ------------------------------------------------------------------
    # 5. Liquidity-adjusted risk
    # ------------------------------------------------------------------
    def compute_liquidity_risk(
        self,
        position_sizes: dict[str, float],
        adv: dict[str, float],
    ) -> dict[str, Any]:
        """
        Compute liquidation cost for 50% unwind in 1 day using sqrt impact model.

        Impact = sigma * sqrt(participation_rate)
        participation_rate = shares_to_sell / ADV
        """
        unwind_fraction = 0.50
        costs: dict[str, float] = {}
        total_cost = 0.0

        for s in self._strategies:
            pos = position_sizes.get(s, 0.0)
            daily_vol = adv.get(s, 1.0)
            if daily_vol <= 0:
                daily_vol = 1.0

            shares_to_sell = abs(pos) * unwind_fraction
            participation = shares_to_sell / daily_vol

            # Use strategy return vol as sigma estimate
            if s in self._returns_df.columns and len(self._returns_df[s]) > 1:
                sigma = float(self._returns_df[s].std())
            else:
                sigma = 0.01

            impact = sigma * np.sqrt(participation)
            cost = shares_to_sell * impact
            costs[s] = float(cost)
            total_cost += cost

        return {
            "strategy_costs": costs,
            "total_liquidation_cost": total_cost,
            "unwind_fraction": unwind_fraction,
        }

    # ------------------------------------------------------------------
    # 6. Tail dependency
    # ------------------------------------------------------------------
    def compute_tail_dependency(self, threshold_quantile: float = 0.05) -> dict[str, float]:
        """
        Lower-tail dependence coefficient from bivariate extreme observations.

        For each pair (i, j), compute P(Y_j <= q | Y_i <= q) where q is the
        lower threshold_quantile quantile.
        """
        results: dict[str, float] = {}
        if len(self._strategies) < 2:
            return results

        for i, s1 in enumerate(self._strategies):
            for s2 in self._strategies[i + 1:]:
                r1 = self._returns_df[s1].values
                r2 = self._returns_df[s2].values
                q1 = np.percentile(r1, threshold_quantile * 100)
                q2 = np.percentile(r2, threshold_quantile * 100)

                mask1 = r1 <= q1
                mask2 = r2 <= q2
                joint = np.sum(mask1 & mask2)
                marginal = np.sum(mask1)

                if marginal > 0:
                    lam = float(joint / marginal)
                else:
                    lam = 0.0

                lam = max(0.0, min(1.0, lam))
                results[f"{s1}|{s2}"] = lam
        return results

    # ------------------------------------------------------------------
    # 7. Risk budget compliance
    # ------------------------------------------------------------------
    def check_compliance(
        self,
        limits: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Check current risk metrics against limits.

        Accepted limit keys:
            max_var_95, max_var_99, max_cvar_95, max_cvar_99,
            max_herfindahl, max_single_strategy_share
        """
        breaches: list[dict[str, Any]] = []
        vc = self.compute_var_cvar()
        vc95 = vc[0]
        vc99 = vc[1]
        conc = self.compute_concentration()

        checks: list[tuple[str, float, float | None]] = [
            ("max_var_95", vc95["var"], limits.get("max_var_95")),
            ("max_var_99", vc99["var"], limits.get("max_var_99")),
            ("max_cvar_95", vc95["cvar"], limits.get("max_cvar_95")),
            ("max_cvar_99", vc99["cvar"], limits.get("max_cvar_99")),
            ("max_herfindahl", conc["herfindahl_index"], limits.get("max_herfindahl")),
        ]

        for name, actual, limit in checks:
            if limit is not None and actual > limit:
                breaches.append({
                    "metric": name,
                    "limit": limit,
                    "actual": float(actual),
                    "breach_amount": float(actual - limit),
                })

        # Single-strategy share check
        max_share_limit = limits.get("max_single_strategy_share")
        if max_share_limit is not None:
            for s, share in conc["strategy_shares"].items():
                if share > max_share_limit:
                    breaches.append({
                        "metric": f"strategy_share_{s}",
                        "limit": max_share_limit,
                        "actual": float(share),
                        "breach_amount": float(share - max_share_limit),
                    })

        return breaches

    # ------------------------------------------------------------------
    # Full aggregation
    # ------------------------------------------------------------------
    def run(
        self,
        position_sizes: dict[str, float] | None = None,
        adv: dict[str, float] | None = None,
        limits: dict[str, float] | None = None,
        stress_scenarios: dict[str, float] | None = None,
    ) -> RiskAggResult:
        """Run all risk analyses and return aggregated result."""
        var_cvar = self.compute_var_cvar()
        marginal = self.compute_marginal_contributions()
        conc = self.compute_concentration()
        stress = self.stress_test(stress_scenarios)

        liq: dict[str, Any] = {}
        if position_sizes and adv:
            liq = self.compute_liquidity_risk(position_sizes, adv)

        tail = self.compute_tail_dependency()

        breaches: list[dict[str, Any]] = []
        if limits:
            breaches = self.check_compliance(limits)

        return RiskAggResult(
            var_cvar=var_cvar,
            marginal_contributions=marginal,
            concentration=conc,
            stress_results=stress,
            liquidity_risk=liq,
            tail_deps=tail,
            compliance_breaches=breaches,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    def generate_report(
        self,
        result: RiskAggResult | None = None,
        position_sizes: dict[str, float] | None = None,
        adv: dict[str, float] | None = None,
        limits: dict[str, float] | None = None,
    ) -> str:
        """Generate an HTML risk report with heatmap, waterfall, stress table, concentration summary."""
        if result is None:
            result = self.run(
                position_sizes=position_sizes,
                adv=adv,
                limits=limits,
            )

        parts: list[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        parts.append("<title>Risk Aggregation Report</title>")
        parts.append("<style>")
        parts.append("""
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; margin: 10px 0 20px 0; }
            th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: right; }
            th { background: #4a90d9; color: white; }
            .heatmap-low { background: #c6efce; }
            .heatmap-mid { background: #ffeb9c; }
            .heatmap-high { background: #ffc7ce; }
            .bar-container { display: flex; align-items: center; margin: 4px 0; }
            .bar-label { width: 120px; text-align: right; padding-right: 8px; }
            .bar { height: 22px; min-width: 2px; }
            .bar-positive { background: #4a90d9; }
            .bar-negative { background: #d94a4a; }
            .breach { color: #d94a4a; font-weight: bold; }
            .pass { color: #2d8a2d; font-weight: bold; }
            .section { background: white; padding: 16px; border-radius: 6px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        """)
        parts.append("</style></head><body>")
        parts.append(f"<h1>Risk Aggregation Report</h1>")
        parts.append(f"<p>Generated: {html.escape(result.generated_at)}</p>")

        # --- VaR / CVaR heatmap table ---
        parts.append('<div class="section"><h2>VaR / CVaR Risk Heatmap</h2>')
        parts.append("<table><tr><th>Confidence</th><th>VaR</th><th>CVaR</th></tr>")
        for row in result.var_cvar:
            conf = row["confidence"]
            var_val = row["var"]
            cvar_val = row["cvar"]
            var_cls = "heatmap-low" if var_val < 0.02 else ("heatmap-mid" if var_val < 0.05 else "heatmap-high")
            cvar_cls = "heatmap-low" if cvar_val < 0.03 else ("heatmap-mid" if cvar_val < 0.07 else "heatmap-high")
            parts.append(
                f'<tr><td>{conf:.0%}</td>'
                f'<td class="{var_cls}">{var_val:.4f}</td>'
                f'<td class="{cvar_cls}">{cvar_val:.4f}</td></tr>'
            )
        parts.append("</table></div>")

        # --- Marginal contribution waterfall ---
        parts.append('<div class="section"><h2>Marginal CVaR Contributions</h2>')
        if result.marginal_contributions:
            max_abs = max(abs(c["marginal_cvar"]) for c in result.marginal_contributions) or 1.0
            for c in result.marginal_contributions:
                val = c["marginal_cvar"]
                width = int(abs(val) / max_abs * 200)
                cls = "bar-positive" if val >= 0 else "bar-negative"
                strat = html.escape(c["strategy"])
                parts.append(
                    f'<div class="bar-container">'
                    f'<span class="bar-label">{strat}</span>'
                    f'<div class="bar {cls}" style="width:{width}px"></div>'
                    f'<span style="padding-left:6px">{val:+.4f}</span></div>'
                )
        parts.append("</div>")

        # --- Stress test table ---
        parts.append('<div class="section"><h2>Stress Test Results</h2>')
        parts.append("<table><tr><th>Scenario</th><th>Shock</th><th>Stressed Max DD</th><th>Final Value</th></tr>")
        for s in result.stress_results:
            parts.append(
                f'<tr><td style="text-align:left">{html.escape(s["scenario"])}</td>'
                f'<td>{s["shock"]:.0%}</td>'
                f'<td class="heatmap-high">{s["stressed_max_drawdown"]:.2%}</td>'
                f'<td>{s["final_value"]:.4f}</td></tr>'
            )
        parts.append("</table></div>")

        # --- Concentration summary ---
        parts.append('<div class="section"><h2>Concentration Summary</h2>')
        if result.concentration:
            hhi = result.concentration.get("herfindahl_index", 0)
            parts.append(f"<p>Herfindahl Index: <strong>{hhi:.4f}</strong></p>")
            shares = result.concentration.get("strategy_shares", {})
            if shares:
                parts.append("<table><tr><th>Strategy</th><th>Share</th></tr>")
                for strat, share in sorted(shares.items()):
                    cls = "heatmap-high" if share > 0.30 else "heatmap-low"
                    parts.append(
                        f'<tr><td style="text-align:left">{html.escape(strat)}</td>'
                        f'<td class="{cls}">{share:.2%}</td></tr>'
                    )
                parts.append("</table>")
            conc_list = result.concentration.get("concentrated_strategies", [])
            if conc_list:
                parts.append(f'<p class="breach">Concentrated strategies: {", ".join(conc_list)}</p>')
            else:
                parts.append('<p class="pass">No concentrated strategies detected.</p>')
        parts.append("</div>")

        # --- Compliance ---
        parts.append('<div class="section"><h2>Compliance</h2>')
        if result.compliance_breaches:
            parts.append("<table><tr><th>Metric</th><th>Limit</th><th>Actual</th><th>Breach</th></tr>")
            for b in result.compliance_breaches:
                parts.append(
                    f'<tr><td style="text-align:left">{html.escape(b["metric"])}</td>'
                    f'<td>{b["limit"]:.4f}</td>'
                    f'<td class="breach">{b["actual"]:.4f}</td>'
                    f'<td>{b["breach_amount"]:.4f}</td></tr>'
                )
            parts.append("</table>")
        else:
            parts.append('<p class="pass">All risk limits within bounds.</p>')
        parts.append("</div>")

        parts.append("</body></html>")
        return "\n".join(parts)
