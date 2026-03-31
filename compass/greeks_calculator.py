"""
Options Greeks calculator with portfolio aggregation and risk limits.

Computes Black-Scholes Greeks (delta, gamma, theta, vega, rho) for
individual options and spreads, aggregates across a portfolio of
experiments, tracks Greeks evolution over time (theta decay, gamma
exposure), runs scenario analysis across underlying/vol/time, and
enforces configurable risk limits.

Generates an HTML report at reports/greeks_dashboard.html with Greeks
summary cards, decay curves, risk limit status, and scenario matrix.

Usage::

    from compass.greeks_calculator import GreeksCalculator
    calc = GreeksCalculator(positions)
    results = calc.analyze()
    calc.generate_report("reports/greeks_dashboard.html")
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
DEFAULT_OUTPUT = ROOT / "reports" / "greeks_dashboard.html"

_SQRT2PI = math.sqrt(2.0 * math.pi)
_TRADING_DAYS = 252.0


# ── Black-Scholes core ─────────────────────────────────────────────────


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _d1d2(S: float, K: float, T: float, sigma: float, r: float) -> Tuple[float, float]:
    """Compute d1, d2 for Black-Scholes."""
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class OptionGreeks:
    """Greeks for a single option leg."""
    delta: float
    gamma: float
    theta: float       # per calendar day
    vega: float        # per 1% IV move
    rho: float         # per 1% rate move
    price: float


@dataclass
class SpreadGreeks:
    """Net Greeks for a spread (short leg − long leg)."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    net_price: float


@dataclass
class Position:
    """A single option or spread position in the portfolio."""
    experiment: str        # e.g. "EXP-400"
    option_type: str       # "call" or "put"
    direction: str         # "long" or "short"
    strike: float
    underlying_price: float
    iv: float              # annualized implied vol
    dte: float             # days to expiration
    contracts: int = 1
    rate: float = 0.045
    # For spreads: second leg
    spread_strike: Optional[float] = None   # long leg strike for credit spreads


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for the entire portfolio."""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    by_experiment: Dict[str, SpreadGreeks]
    n_positions: int


@dataclass
class RiskLimit:
    """A configurable risk limit with current status."""
    metric: str
    limit: float
    current: float
    breached: bool
    utilization: float     # current / limit


@dataclass
class ScenarioResult:
    """Greeks at a single scenario point."""
    underlying_shift: float   # % change from current
    vol_shift: float          # absolute IV change
    dte_shift: float          # days subtracted
    delta: float
    gamma: float
    theta: float
    vega: float
    pnl: float


@dataclass
class DecayPoint:
    """Greeks at a point along a time-decay curve."""
    dte: float
    delta: float
    gamma: float
    theta: float
    vega: float
    price: float


# ── Greeks computation ──────────────────────────────────────────────────


def compute_option_greeks(
    S: float, K: float, T: float, sigma: float, r: float,
    option_type: str,
) -> OptionGreeks:
    """Compute Black-Scholes Greeks for a single European option.

    Parameters
    ----------
    S : underlying price
    K : strike price
    T : time to expiry in years (>0)
    sigma : annualized implied volatility (>0)
    r : risk-free rate
    option_type : "call" or "put"
    """
    if T <= 1e-10 or sigma <= 1e-10 or S <= 0 or K <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return OptionGreeks(
            delta=1.0 if option_type == "call" and S > K else (-1.0 if option_type == "put" and K > S else 0.0),
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0, price=intrinsic,
        )

    d1, d2 = _d1d2(S, K, T, sigma, r)
    sqrt_t = math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)
    disc = math.exp(-r * T)

    if option_type == "call":
        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)
        price = S * nd1 - K * disc * nd2
        delta = nd1
        rho_val = K * T * disc * nd2 / 100.0
    else:
        nd1 = _norm_cdf(-d1)
        nd2 = _norm_cdf(-d2)
        price = K * disc * nd2 - S * nd1
        delta = -nd1
        rho_val = -K * T * disc * nd2 / 100.0

    gamma = pdf_d1 / (S * sigma * sqrt_t)
    theta = (
        -(S * pdf_d1 * sigma) / (2.0 * sqrt_t)
        - r * K * disc * (_norm_cdf(d2) if option_type == "call" else _norm_cdf(-d2))
        * (1 if option_type == "call" else -1)
    ) / _TRADING_DAYS
    vega = S * pdf_d1 * sqrt_t / 100.0

    return OptionGreeks(
        delta=delta, gamma=gamma, theta=theta,
        vega=vega, rho=rho_val, price=price,
    )


def compute_spread_greeks(position: Position) -> SpreadGreeks:
    """Compute net Greeks for a position (single option or spread)."""
    T = max(position.dte / 365.0, 1e-10)
    sign = -1.0 if position.direction == "short" else 1.0
    mult = position.contracts * 100.0  # standard 100 shares/contract

    main = compute_option_greeks(
        position.underlying_price, position.strike, T,
        position.iv, position.rate, position.option_type,
    )
    net = SpreadGreeks(
        delta=sign * main.delta * mult,
        gamma=sign * main.gamma * mult,
        theta=sign * main.theta * mult,
        vega=sign * main.vega * mult,
        rho=sign * main.rho * mult,
        net_price=sign * main.price * mult,
    )

    if position.spread_strike is not None:
        # Spread: main leg + opposite long leg
        hedge = compute_option_greeks(
            position.underlying_price, position.spread_strike, T,
            position.iv, position.rate, position.option_type,
        )
        hedge_sign = -sign  # opposite direction
        net = SpreadGreeks(
            delta=net.delta + hedge_sign * hedge.delta * mult,
            gamma=net.gamma + hedge_sign * hedge.gamma * mult,
            theta=net.theta + hedge_sign * hedge.theta * mult,
            vega=net.vega + hedge_sign * hedge.vega * mult,
            rho=net.rho + hedge_sign * hedge.rho * mult,
            net_price=net.net_price + hedge_sign * hedge.price * mult,
        )
    return net


# ── Calculator ──────────────────────────────────────────────────────────


class GreeksCalculator:
    """Portfolio-level Greeks calculator with scenario analysis and risk limits."""

    def __init__(
        self,
        positions: List[Position],
        risk_limits: Optional[Dict[str, float]] = None,
    ) -> None:
        self.positions = list(positions)
        self.risk_limits_config = risk_limits or {
            "delta": 500.0,
            "gamma": 100.0,
            "vega": 1000.0,
        }

        # Results
        self.position_greeks: List[Tuple[Position, SpreadGreeks]] = []
        self.portfolio: Optional[PortfolioGreeks] = None
        self.risk_limits: List[RiskLimit] = []
        self.scenarios: List[ScenarioResult] = []
        self.decay_curves: Dict[str, List[DecayPoint]] = {}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **kwargs: Any) -> "GreeksCalculator":
        """Build from a DataFrame with position columns."""
        positions = []
        for _, row in df.iterrows():
            pos = Position(
                experiment=str(row.get("experiment", "unknown")),
                option_type=str(row.get("option_type", "put")),
                direction=str(row.get("direction", "short")),
                strike=float(row["strike"]),
                underlying_price=float(row["underlying_price"]),
                iv=float(row["iv"]),
                dte=float(row["dte"]),
                contracts=int(row.get("contracts", 1)),
                rate=float(row.get("rate", 0.045)),
                spread_strike=float(row["spread_strike"]) if pd.notna(row.get("spread_strike")) else None,
            )
            positions.append(pos)
        return cls(positions, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """Run full analysis."""
        self.position_greeks = self._compute_all_greeks()
        self.portfolio = self._aggregate_portfolio()
        self.risk_limits = self._check_risk_limits()
        self.scenarios = self._scenario_analysis()
        self.decay_curves = self._theta_decay_curves()
        return {
            "position_greeks": self.position_greeks,
            "portfolio": self.portfolio,
            "risk_limits": self.risk_limits,
            "scenarios": self.scenarios,
            "decay_curves": self.decay_curves,
        }

    # ── Position Greeks ─────────────────────────────────────────────────

    def _compute_all_greeks(self) -> List[Tuple[Position, SpreadGreeks]]:
        return [(pos, compute_spread_greeks(pos)) for pos in self.positions]

    # ── Portfolio aggregation ───────────────────────────────────────────

    def _aggregate_portfolio(self) -> PortfolioGreeks:
        """Aggregate Greeks across all positions, grouped by experiment."""
        by_exp: Dict[str, SpreadGreeks] = {}
        totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

        for pos, sg in self.position_greeks:
            exp = pos.experiment
            if exp not in by_exp:
                by_exp[exp] = SpreadGreeks(0, 0, 0, 0, 0, 0)
            cur = by_exp[exp]
            by_exp[exp] = SpreadGreeks(
                delta=cur.delta + sg.delta,
                gamma=cur.gamma + sg.gamma,
                theta=cur.theta + sg.theta,
                vega=cur.vega + sg.vega,
                rho=cur.rho + sg.rho,
                net_price=cur.net_price + sg.net_price,
            )
            totals["delta"] += sg.delta
            totals["gamma"] += sg.gamma
            totals["theta"] += sg.theta
            totals["vega"] += sg.vega
            totals["rho"] += sg.rho

        return PortfolioGreeks(
            total_delta=totals["delta"],
            total_gamma=totals["gamma"],
            total_theta=totals["theta"],
            total_vega=totals["vega"],
            total_rho=totals["rho"],
            by_experiment=by_exp,
            n_positions=len(self.positions),
        )

    # ── Risk limits ─────────────────────────────────────────────────────

    def _check_risk_limits(self) -> List[RiskLimit]:
        """Check portfolio Greeks against configured risk limits."""
        if self.portfolio is None:
            return []
        limits: List[RiskLimit] = []
        mapping = {
            "delta": abs(self.portfolio.total_delta),
            "gamma": abs(self.portfolio.total_gamma),
            "vega": abs(self.portfolio.total_vega),
        }
        for metric, limit_val in self.risk_limits_config.items():
            current = mapping.get(metric, 0.0)
            util = current / limit_val if limit_val > 0 else 0.0
            limits.append(RiskLimit(
                metric=metric, limit=limit_val, current=current,
                breached=current > limit_val, utilization=util,
            ))
        return limits

    # ── Scenario analysis ───────────────────────────────────────────────

    def _scenario_analysis(
        self,
        underlying_shifts: Optional[Sequence[float]] = None,
        vol_shifts: Optional[Sequence[float]] = None,
        dte_shifts: Optional[Sequence[float]] = None,
    ) -> List[ScenarioResult]:
        """Sweep underlying/vol/time and compute portfolio Greeks at each point."""
        if underlying_shifts is None:
            underlying_shifts = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]
        if vol_shifts is None:
            vol_shifts = [-0.10, -0.05, 0.0, 0.05, 0.10]
        if dte_shifts is None:
            dte_shifts = [0, 5, 10, 20]

        base_price = self.portfolio.total_delta if self.portfolio else 0
        results: List[ScenarioResult] = []

        for u_shift in underlying_shifts:
            for v_shift in vol_shifts:
                for d_shift in dte_shifts:
                    tot = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "pnl": 0.0}
                    for pos, base_sg in self.position_greeks:
                        shifted = Position(
                            experiment=pos.experiment,
                            option_type=pos.option_type,
                            direction=pos.direction,
                            strike=pos.strike,
                            underlying_price=pos.underlying_price * (1.0 + u_shift),
                            iv=max(pos.iv + v_shift, 0.01),
                            dte=max(pos.dte - d_shift, 0.01),
                            contracts=pos.contracts,
                            rate=pos.rate,
                            spread_strike=pos.spread_strike,
                        )
                        sg = compute_spread_greeks(shifted)
                        tot["delta"] += sg.delta
                        tot["gamma"] += sg.gamma
                        tot["theta"] += sg.theta
                        tot["vega"] += sg.vega
                        tot["pnl"] += sg.net_price - base_sg.net_price
                    results.append(ScenarioResult(
                        underlying_shift=u_shift, vol_shift=v_shift,
                        dte_shift=d_shift,
                        delta=tot["delta"], gamma=tot["gamma"],
                        theta=tot["theta"], vega=tot["vega"],
                        pnl=tot["pnl"],
                    ))
        return results

    # ── Theta decay curves ──────────────────────────────────────────────

    def _theta_decay_curves(self) -> Dict[str, List[DecayPoint]]:
        """Compute Greeks evolution as time passes for each experiment."""
        curves: Dict[str, List[DecayPoint]] = {}
        max_dte = max((p.dte for p in self.positions), default=45)
        dte_range = np.arange(max_dte, 0, -1)

        for exp in {p.experiment for p in self.positions}:
            exp_positions = [p for p in self.positions if p.experiment == exp]
            points: List[DecayPoint] = []
            for dte in dte_range:
                totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "price": 0.0}
                for pos in exp_positions:
                    if dte > pos.dte:
                        continue
                    shifted = Position(
                        experiment=pos.experiment,
                        option_type=pos.option_type,
                        direction=pos.direction,
                        strike=pos.strike,
                        underlying_price=pos.underlying_price,
                        iv=pos.iv,
                        dte=dte,
                        contracts=pos.contracts,
                        rate=pos.rate,
                        spread_strike=pos.spread_strike,
                    )
                    sg = compute_spread_greeks(shifted)
                    totals["delta"] += sg.delta
                    totals["gamma"] += sg.gamma
                    totals["theta"] += sg.theta
                    totals["vega"] += sg.vega
                    totals["price"] += sg.net_price
                points.append(DecayPoint(
                    dte=float(dte), delta=totals["delta"], gamma=totals["gamma"],
                    theta=totals["theta"], vega=totals["vega"], price=totals["price"],
                ))
            curves[exp] = points
        return curves

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate HTML report. Runs analyze() if not yet run."""
        if self.portfolio is None:
            self.analyze()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Report written to %s", out)
        return str(out.resolve())

    # ── Charts ──────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        charts: Dict[str, str] = {}
        charts["decay"] = self._chart_decay_curves()
        charts["risk_limits"] = self._chart_risk_limits()
        charts["scenario"] = self._chart_scenario_matrix()
        charts["gamma_exposure"] = self._chart_gamma_exposure()
        return charts

    def _chart_decay_curves(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.decay_curves:
            return ""
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        metrics = [("theta", "Theta ($/day)"), ("delta", "Delta"),
                   ("gamma", "Gamma"), ("vega", "Vega ($/1% IV)")]
        colors = ["#3b82f6", "#16a34a", "#f59e0b", "#dc2626", "#8b5cf6"]
        for ax, (metric, label) in zip(axes.flat, metrics):
            for k, (exp, points) in enumerate(self.decay_curves.items()):
                xs = [p.dte for p in points]
                ys = [getattr(p, metric) for p in points]
                ax.plot(xs, ys, label=exp, color=colors[k % len(colors)], lw=1.2)
            ax.set_xlabel("DTE")
            ax.set_ylabel(label)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7)
        fig.suptitle("Greeks Decay Curves", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_risk_limits(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.risk_limits:
            return ""
        fig, ax = plt.subplots(figsize=(6, 3))
        names = [rl.metric.upper() for rl in self.risk_limits]
        utils = [min(rl.utilization, 1.5) for rl in self.risk_limits]
        colors = ["#dc2626" if rl.breached else "#16a34a" if rl.utilization < 0.7 else "#f59e0b"
                  for rl in self.risk_limits]
        ax.barh(names, utils, color=colors, alpha=0.85, edgecolor="white")
        ax.axvline(1.0, color="#dc2626", lw=1.5, ls="--", label="Limit")
        ax.set_xlabel("Utilization (current / limit)")
        ax.set_title("Risk Limit Status", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_scenario_matrix(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.scenarios:
            return ""
        # Filter to dte_shift=0 for the underlying×vol matrix
        subset = [s for s in self.scenarios if s.dte_shift == 0]
        if not subset:
            return ""
        u_shifts = sorted(set(s.underlying_shift for s in subset))
        v_shifts = sorted(set(s.vol_shift for s in subset))
        matrix = np.zeros((len(v_shifts), len(u_shifts)))
        for s in subset:
            i = v_shifts.index(s.vol_shift)
            j = u_shifts.index(s.underlying_shift)
            matrix[i, j] = s.pnl

        fig, ax = plt.subplots(figsize=(8, 5))
        vmax = max(abs(matrix.max()), abs(matrix.min()), 1)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(u_shifts)))
        ax.set_xticklabels([f"{u:+.0%}" for u in u_shifts], fontsize=8)
        ax.set_yticks(range(len(v_shifts)))
        ax.set_yticklabels([f"{v:+.0%}" for v in v_shifts], fontsize=8)
        ax.set_xlabel("Underlying Shift")
        ax.set_ylabel("IV Shift")
        for i in range(len(v_shifts)):
            for j in range(len(u_shifts)):
                ax.text(j, i, f"${matrix[i, j]:,.0f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(matrix[i, j]) > vmax * 0.6 else "black")
        fig.colorbar(im, shrink=0.8, label="P&L ($)")
        ax.set_title("Scenario P&L Matrix (DTE unchanged)", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_gamma_exposure(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.position_greeks:
            return ""
        fig, ax = plt.subplots(figsize=(7, 4))
        strikes = [pos.strike for pos, _ in self.position_greeks]
        gammas = [sg.gamma for _, sg in self.position_greeks]
        labels = [pos.experiment for pos, _ in self.position_greeks]
        exp_colors = {}
        palette = ["#3b82f6", "#16a34a", "#f59e0b", "#dc2626", "#8b5cf6"]
        for pos, _ in self.position_greeks:
            if pos.experiment not in exp_colors:
                exp_colors[pos.experiment] = palette[len(exp_colors) % len(palette)]
        colors = [exp_colors[pos.experiment] for pos, _ in self.position_greeks]
        ax.bar(range(len(strikes)), gammas, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(strikes)))
        ax.set_xticklabels([f"{s:.0f}" for s in strikes], fontsize=7, rotation=45)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Gamma Exposure")
        ax.set_title("Gamma Exposure by Strike", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=e) for e, c in exp_colors.items()]
        ax.legend(handles=handles, fontsize=8)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        pf = self.portfolio or PortfolioGreeks(0, 0, 0, 0, 0, {}, 0)

        # Risk limit rows
        risk_rows = ""
        for rl in self.risk_limits:
            cls = "bad" if rl.breached else "good" if rl.utilization < 0.7 else "warn"
            status = "BREACHED" if rl.breached else "OK"
            risk_rows += (
                f'<tr><td>{rl.metric.upper()}</td>'
                f'<td>{rl.current:.1f}</td>'
                f'<td>{rl.limit:.1f}</td>'
                f'<td class="{cls}">{rl.utilization:.0%}</td>'
                f'<td class="{cls}">{status}</td></tr>\n'
            )

        # Position Greeks table
        pos_rows = ""
        for pos, sg in self.position_greeks:
            spread_str = f"/{pos.spread_strike:.0f}" if pos.spread_strike else ""
            pos_rows += (
                f'<tr><td>{pos.experiment}</td>'
                f'<td>{pos.direction} {pos.option_type} {pos.strike:.0f}{spread_str}</td>'
                f'<td>{pos.contracts}</td>'
                f'<td>{sg.delta:.2f}</td>'
                f'<td>{sg.gamma:.4f}</td>'
                f'<td>{sg.theta:.2f}</td>'
                f'<td>{sg.vega:.2f}</td>'
                f'<td>{sg.rho:.2f}</td></tr>\n'
            )

        # Experiment aggregation
        exp_rows = ""
        for exp, sg in pf.by_experiment.items():
            exp_rows += (
                f'<tr><td>{exp}</td>'
                f'<td>{sg.delta:.2f}</td>'
                f'<td>{sg.gamma:.4f}</td>'
                f'<td>{sg.theta:.2f}</td>'
                f'<td>{sg.vega:.2f}</td>'
                f'<td>${sg.net_price:,.0f}</td></tr>\n'
            )

        breached_count = sum(1 for rl in self.risk_limits if rl.breached)
        risk_cls = "bad" if breached_count > 0 else "good"

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Options Greeks Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .warn {{ color: #f59e0b; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
  td:first-child {{ text-align: left; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Options Greeks Dashboard</h1>
<div class="meta">{pf.n_positions} positions &middot; {len(pf.by_experiment)} experiments &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{pf.total_delta:.1f}</div><div class="label">Portfolio Delta</div></div>
  <div class="kpi"><div class="value">{pf.total_gamma:.3f}</div><div class="label">Portfolio Gamma</div></div>
  <div class="kpi"><div class="value">${pf.total_theta:.0f}</div><div class="label">Daily Theta</div></div>
  <div class="kpi"><div class="value">${pf.total_vega:.0f}</div><div class="label">Vega Exposure</div></div>
  <div class="kpi"><div class="value {risk_cls}">{breached_count} / {len(self.risk_limits)}</div><div class="label">Limits Breached</div></div>
</div>

<h2>1. Risk Limit Status</h2>
{_img("risk_limits")}
<table>
<thead><tr><th>Metric</th><th>Current</th><th>Limit</th><th>Utilization</th><th>Status</th></tr></thead>
<tbody>{risk_rows}</tbody>
</table>

<h2>2. Position Greeks</h2>
<table>
<thead><tr><th>Experiment</th><th>Position</th><th>Contracts</th><th>&Delta;</th><th>&Gamma;</th><th>&Theta;</th><th>Vega</th><th>&rho;</th></tr></thead>
<tbody>{pos_rows}</tbody>
</table>

<h2>3. Experiment Aggregation</h2>
<table>
<thead><tr><th>Experiment</th><th>&Delta;</th><th>&Gamma;</th><th>&Theta;</th><th>Vega</th><th>Net Value</th></tr></thead>
<tbody>{exp_rows}</tbody>
</table>

<h2>4. Theta Decay Curves</h2>
{_img("decay")}

<h2>5. Gamma Exposure</h2>
{_img("gamma_exposure")}

<h2>6. Scenario Analysis</h2>
{_img("scenario")}

<footer>Generated by <code>compass/greeks_calculator.py</code></footer>
</body></html>"""
        return html
