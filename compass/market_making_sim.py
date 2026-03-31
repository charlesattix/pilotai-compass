"""
Market-making simulator with Avellaneda-Stoikov optimal quoting.

Implements the Avellaneda-Stoikov (2008) model for optimal bid/ask
placement, bid-ask spread optimisation, inventory risk management,
adverse-selection modelling, PnL simulation with realistic Poisson
order flow, and optimal quote-depth analysis.

Generates an HTML report at reports/market_making_sim.html with
inventory path, PnL curves, and spread evolution charts.

Usage::

    from compass.market_making_sim import MarketMakingSim
    sim = MarketMakingSim(mid_prices)
    results = sim.run()
    sim.generate_report()
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "market_making_sim.html"


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class ASParams:
    """Avellaneda-Stoikov model parameters."""
    gamma: float = 0.1        # risk-aversion
    sigma: float = 0.3        # volatility (annualised)
    k: float = 1.5            # order-arrival intensity shape
    A: float = 140.0          # order-arrival scaling
    T: float = 1.0            # terminal time (fraction of day)
    dt: float = 1 / 390       # time-step (1 minute of 390-min day)
    max_inventory: int = 10   # hard inventory cap per side
    rebate: float = 0.0       # maker rebate per share


@dataclass
class QuoteSnapshot:
    """Quotes and state at a single time-step."""
    t: float
    mid: float
    bid: float
    ask: float
    spread: float
    reservation: float
    inventory: int
    pnl: float
    cash: float
    mark_to_market: float


@dataclass
class Fill:
    """An executed fill."""
    t: float
    side: str            # "buy" or "sell"
    price: float
    mid_at_fill: float
    inventory_after: int
    adverse: bool        # was the fill adversely selected?


@dataclass
class SimResult:
    """Complete simulation result."""
    snapshots: List[QuoteSnapshot]
    fills: List[Fill]
    total_pnl: float
    final_inventory: int
    n_fills: int
    n_adverse: int
    adverse_pct: float
    sharpe: float
    max_drawdown: float
    avg_spread: float
    avg_inventory: float
    turnover: int


@dataclass
class DepthAnalysis:
    """Optimal quote-depth analysis result."""
    depth: float
    avg_pnl: float
    sharpe: float
    avg_spread: float
    fill_rate: float
    adverse_pct: float


# ── Avellaneda-Stoikov core ─────────────────────────────────────────────


def reservation_price(mid: float, q: int, gamma: float, sigma: float,
                      T_remaining: float) -> float:
    """Compute the reservation (indifference) price.

    r = s - q * gamma * sigma^2 * (T - t)
    """
    return mid - q * gamma * sigma ** 2 * T_remaining


def optimal_spread(gamma: float, sigma: float, T_remaining: float,
                   k: float, A: float) -> float:
    """Compute the optimal spread (δ* on each side).

    δ* = gamma * σ² * (T-t) + (2/gamma) * ln(1 + gamma/k)
    """
    return gamma * sigma ** 2 * T_remaining + (2.0 / gamma) * math.log(1 + gamma / k)


def fill_probability(delta: float, A: float, k: float) -> float:
    """Probability of fill at distance δ from mid.

    λ(δ) = A * exp(-k * δ)
    """
    return A * math.exp(-k * delta)


# ── Simulator ───────────────────────────────────────────────────────────


class MarketMakingSim:
    """Full market-making simulation with Avellaneda-Stoikov quoting."""

    def __init__(
        self,
        mid_prices: pd.Series,
        params: Optional[ASParams] = None,
        adverse_fraction: float = 0.15,
        adverse_move: float = 0.002,
        seed: int = 42,
    ) -> None:
        self.mid_prices = mid_prices.values.astype(float)
        self.index = mid_prices.index if hasattr(mid_prices, 'index') else range(len(mid_prices))
        self.params = params or ASParams()
        self.adverse_fraction = adverse_fraction
        self.adverse_move = adverse_move
        self.rng = np.random.RandomState(seed)

        # Results
        self.result: Optional[SimResult] = None

    @classmethod
    def from_random_walk(
        cls, n_steps: int = 390, start: float = 100.0,
        sigma: float = 0.3, seed: int = 42, **kwargs: Any,
    ) -> "MarketMakingSim":
        """Create sim with synthetic mid-price random walk."""
        rng = np.random.RandomState(seed)
        dt = 1.0 / n_steps
        returns = rng.normal(0, sigma * math.sqrt(dt), n_steps)
        prices = start * np.exp(returns.cumsum())
        index = pd.date_range("2024-06-03 09:30", periods=n_steps, freq="1min")
        return cls(pd.Series(prices, index=index), seed=seed, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> SimResult:
        """Run the full simulation."""
        p = self.params
        n = len(self.mid_prices)
        dt = p.dt

        inventory = 0
        cash = 0.0
        snapshots: List[QuoteSnapshot] = []
        fills: List[Fill] = []
        pnl_series: List[float] = []

        for i in range(n):
            t = i * dt
            T_rem = max(p.T - t, dt)
            mid = self.mid_prices[i]

            # Reservation price and optimal spread
            res = reservation_price(mid, inventory, p.gamma, p.sigma, T_rem)
            delta = optimal_spread(p.gamma, p.sigma, T_rem, p.k, p.A)
            half = delta / 2.0

            bid = res - half
            ask = res + half
            spread = ask - bid

            # Mark-to-market
            mtm = cash + inventory * mid
            pnl_series.append(mtm)

            snapshots.append(QuoteSnapshot(
                t=t, mid=mid, bid=bid, ask=ask, spread=spread,
                reservation=res, inventory=inventory, pnl=mtm,
                cash=cash, mark_to_market=mtm,
            ))

            # Simulate fills (Poisson arrival)
            bid_dist = mid - bid
            ask_dist = ask - mid

            bid_lambda = fill_probability(max(bid_dist, 0.001), p.A, p.k) * dt
            ask_lambda = fill_probability(max(ask_dist, 0.001), p.A, p.k) * dt

            # Buy fill (someone hits our bid)
            if self.rng.random() < min(bid_lambda, 0.99) and inventory < p.max_inventory:
                adverse = self.rng.random() < self.adverse_fraction
                fill_price = bid
                if adverse:
                    fill_price = bid * (1 - self.adverse_move)
                cash -= fill_price
                inventory += 1
                fills.append(Fill(
                    t=t, side="buy", price=fill_price,
                    mid_at_fill=mid, inventory_after=inventory,
                    adverse=adverse,
                ))

            # Sell fill (someone lifts our ask)
            if self.rng.random() < min(ask_lambda, 0.99) and inventory > -p.max_inventory:
                adverse = self.rng.random() < self.adverse_fraction
                fill_price = ask
                if adverse:
                    fill_price = ask * (1 + self.adverse_move)
                cash += fill_price
                inventory -= 1
                fills.append(Fill(
                    t=t, side="sell", price=fill_price,
                    mid_at_fill=mid, inventory_after=inventory,
                    adverse=adverse,
                ))

        # Final PnL
        final_mid = self.mid_prices[-1]
        total_pnl = cash + inventory * final_mid
        n_adverse = sum(1 for f in fills if f.adverse)

        # Sharpe
        pnl_arr = np.array(pnl_series)
        pnl_diff = np.diff(pnl_arr) if len(pnl_arr) > 1 else np.array([0.0])
        sharpe = float(pnl_diff.mean() / pnl_diff.std() * math.sqrt(252 * 390)) if pnl_diff.std() > 0 else 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(pnl_arr)
        dd = (cummax - pnl_arr)
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        # Averages
        avg_spread = float(np.mean([s.spread for s in snapshots]))
        avg_inv = float(np.mean([abs(s.inventory) for s in snapshots]))
        turnover = sum(1 for f in fills)

        self.result = SimResult(
            snapshots=snapshots, fills=fills,
            total_pnl=total_pnl, final_inventory=inventory,
            n_fills=len(fills), n_adverse=n_adverse,
            adverse_pct=n_adverse / max(len(fills), 1),
            sharpe=sharpe, max_drawdown=max_dd,
            avg_spread=avg_spread, avg_inventory=avg_inv,
            turnover=turnover,
        )
        return self.result

    def analyze_depth(
        self, depths: Optional[List[float]] = None, n_runs: int = 5,
    ) -> List[DepthAnalysis]:
        """Sweep quote depth (gamma) to find optimal setting."""
        depths = depths or [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        results: List[DepthAnalysis] = []
        for gamma in depths:
            pnls, sharpes, spreads, fill_rates, adverse_pcts = [], [], [], [], []
            for run in range(n_runs):
                params = ASParams(
                    gamma=gamma, sigma=self.params.sigma,
                    k=self.params.k, A=self.params.A,
                    T=self.params.T, dt=self.params.dt,
                    max_inventory=self.params.max_inventory,
                )
                sim = MarketMakingSim(
                    pd.Series(self.mid_prices),
                    params=params,
                    adverse_fraction=self.adverse_fraction,
                    adverse_move=self.adverse_move,
                    seed=self.rng.randint(0, 100000),
                )
                res = sim.run()
                pnls.append(res.total_pnl)
                sharpes.append(res.sharpe)
                spreads.append(res.avg_spread)
                fill_rates.append(res.n_fills / max(len(self.mid_prices), 1))
                adverse_pcts.append(res.adverse_pct)

            results.append(DepthAnalysis(
                depth=gamma,
                avg_pnl=float(np.mean(pnls)),
                sharpe=float(np.mean(sharpes)),
                avg_spread=float(np.mean(spreads)),
                fill_rate=float(np.mean(fill_rates)),
                adverse_pct=float(np.mean(adverse_pcts)),
            ))
        return sorted(results, key=lambda d: -d.sharpe)

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.result is None:
            self.run()
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
        charts["pnl_inventory"] = self._chart_pnl_inventory()
        charts["spread_evolution"] = self._chart_spread_evolution()
        charts["fills"] = self._chart_fills()
        return charts

    def _chart_pnl_inventory(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.result:
            return ""
        snaps = self.result.snapshots
        xs = list(range(len(snaps)))
        pnl = [s.pnl for s in snaps]
        inv = [s.inventory for s in snaps]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(xs, pnl, color="#16a34a", lw=0.9)
        ax1.fill_between(xs, pnl, alpha=0.1, color="#16a34a")
        ax1.set_ylabel("Mark-to-Market P&L")
        ax1.set_title("P&L Path", fontsize=11)
        ax1.grid(True, alpha=0.2)

        colors = ["#dc2626" if i < 0 else "#16a34a" if i > 0 else "#64748b" for i in inv]
        ax2.bar(xs, inv, color=colors, alpha=0.7, width=1.0)
        ax2.axhline(0, color="black", lw=0.5)
        ax2.set_ylabel("Inventory")
        ax2.set_xlabel("Time Step")
        ax2.set_title("Inventory Path", fontsize=11)
        ax2.grid(True, alpha=0.2)

        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_spread_evolution(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.result:
            return ""
        snaps = self.result.snapshots
        xs = list(range(len(snaps)))
        spreads = [s.spread for s in snaps]
        mids = [s.mid for s in snaps]
        bids = [s.bid for s in snaps]
        asks = [s.ask for s in snaps]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(xs, mids, color="#334155", lw=0.8, label="Mid")
        ax1.fill_between(xs, bids, asks, alpha=0.15, color="#3b82f6", label="Bid-Ask")
        ax1.set_ylabel("Price")
        ax1.set_title("Quote Evolution", fontsize=11)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)

        ax2.plot(xs, spreads, color="#f59e0b", lw=0.8)
        ax2.set_ylabel("Spread")
        ax2.set_xlabel("Time Step")
        ax2.set_title("Spread Evolution", fontsize=11)
        ax2.grid(True, alpha=0.2)

        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_fills(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.result or not self.result.fills:
            return ""
        fills = self.result.fills
        buys = [f for f in fills if f.side == "buy"]
        sells = [f for f in fills if f.side == "sell"]

        fig, ax = plt.subplots(figsize=(10, 4))
        snaps = self.result.snapshots
        xs = list(range(len(snaps)))
        mids = [s.mid for s in snaps]
        ax.plot(xs, mids, color="#334155", lw=0.6, alpha=0.5, label="Mid")

        # Map fill time to index
        dt = self.params.dt
        for f in buys:
            idx = min(int(f.t / dt), len(xs) - 1)
            c = "#dc2626" if f.adverse else "#16a34a"
            ax.scatter(idx, f.price, marker="^", s=15, color=c, zorder=5)
        for f in sells:
            idx = min(int(f.t / dt), len(xs) - 1)
            c = "#dc2626" if f.adverse else "#3b82f6"
            ax.scatter(idx, f.price, marker="v", s=15, color=c, zorder=5)

        ax.set_ylabel("Price")
        ax.set_xlabel("Time Step")
        ax.set_title(f"Fills: {len(buys)} buys, {len(sells)} sells ({self.result.n_adverse} adverse)", fontsize=11)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        r = self.result or SimResult([], [], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        p = self.params

        pnl_cls = "good" if r.total_pnl >= 0 else "bad"
        adv_cls = "bad" if r.adverse_pct > 0.2 else "good" if r.adverse_pct < 0.1 else ""

        # Fill summary table
        buy_fills = [f for f in r.fills if f.side == "buy"]
        sell_fills = [f for f in r.fills if f.side == "sell"]
        avg_buy = float(np.mean([f.price for f in buy_fills])) if buy_fills else 0
        avg_sell = float(np.mean([f.price for f in sell_fills])) if sell_fills else 0
        avg_edge = avg_sell - avg_buy if buy_fills and sell_fills else 0

        # Recent fills
        fill_rows = ""
        for f in r.fills[-30:]:
            cls = "bad" if f.adverse else ""
            fill_rows += (
                f'<tr><td>{f.t:.4f}</td><td>{f.side}</td>'
                f'<td>{f.price:.4f}</td><td>{f.mid_at_fill:.4f}</td>'
                f'<td>{f.inventory_after}</td>'
                f'<td class="{cls}">{"Yes" if f.adverse else "No"}</td></tr>\n'
            )
        if not fill_rows:
            fill_rows = '<tr><td colspan="6" style="text-align:center;color:#64748b">No fills</td></tr>'

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Market Making Simulation</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
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

<h1>Market Making Simulation</h1>
<div class="meta">{len(r.snapshots)} steps &middot; &gamma;={p.gamma} &sigma;={p.sigma} k={p.k} &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value {pnl_cls}">${r.total_pnl:,.2f}</div><div class="label">Total P&L</div></div>
  <div class="kpi"><div class="value">{r.sharpe:.2f}</div><div class="label">Sharpe</div></div>
  <div class="kpi"><div class="value">{r.n_fills}</div><div class="label">Fills</div></div>
  <div class="kpi"><div class="value {adv_cls}">{r.adverse_pct:.0%}</div><div class="label">Adverse %</div></div>
  <div class="kpi"><div class="value">{r.avg_spread:.4f}</div><div class="label">Avg Spread</div></div>
  <div class="kpi"><div class="value">{r.avg_inventory:.1f}</div><div class="label">Avg |Inventory|</div></div>
  <div class="kpi"><div class="value">${r.max_drawdown:,.2f}</div><div class="label">Max Drawdown</div></div>
</div>

<h2>1. P&L &amp; Inventory</h2>
{_img("pnl_inventory")}

<h2>2. Quote &amp; Spread Evolution</h2>
{_img("spread_evolution")}

<h2>3. Fill Analysis</h2>
{_img("fills")}
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Buy fills</td><td>{len(buy_fills)}</td></tr>
<tr><td>Sell fills</td><td>{len(sell_fills)}</td></tr>
<tr><td>Avg buy price</td><td>{avg_buy:.4f}</td></tr>
<tr><td>Avg sell price</td><td>{avg_sell:.4f}</td></tr>
<tr><td>Avg edge per round-trip</td><td class="{"good" if avg_edge > 0 else "bad"}">{avg_edge:.4f}</td></tr>
<tr><td>Adverse fills</td><td class="{adv_cls}">{r.n_adverse} ({r.adverse_pct:.0%})</td></tr>
<tr><td>Final inventory</td><td>{r.final_inventory}</td></tr>
</tbody>
</table>

<h2>4. Recent Fills</h2>
<table>
<thead><tr><th>Time</th><th>Side</th><th>Price</th><th>Mid</th><th>Inv After</th><th>Adverse</th></tr></thead>
<tbody>{fill_rows}</tbody>
</table>

<h2>5. Model Parameters</h2>
<table>
<thead><tr><th>Parameter</th><th>Value</th><th>Description</th></tr></thead>
<tbody>
<tr><td>&gamma;</td><td>{p.gamma}</td><td>Risk aversion</td></tr>
<tr><td>&sigma;</td><td>{p.sigma}</td><td>Volatility (annualised)</td></tr>
<tr><td>k</td><td>{p.k}</td><td>Order-arrival intensity shape</td></tr>
<tr><td>A</td><td>{p.A}</td><td>Order-arrival scaling</td></tr>
<tr><td>T</td><td>{p.T}</td><td>Terminal time</td></tr>
<tr><td>Max inventory</td><td>{p.max_inventory}</td><td>Hard cap per side</td></tr>
</tbody>
</table>

<footer>Generated by <code>compass/market_making_sim.py</code></footer>
</body></html>"""
        return html
