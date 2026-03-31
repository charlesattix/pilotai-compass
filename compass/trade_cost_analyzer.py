"""
Comprehensive trade cost analyzer for credit spread portfolios.

Computes explicit costs (commissions, fees, taxes), implicit costs
(bid-ask spread, slippage, market impact via Almgren-Chriss model),
opportunity costs (delay cost, missed trades), and provides cost
attribution by strategy/asset/time, cost forecasting for hypothetical
trades, and optimization recommendations (optimal trade size, timing).

Generates an HTML report at reports/trade_costs.html with cost breakdown
waterfall, per-strategy comparison, and cost trend charts.

Usage::

    from compass.trade_cost_analyzer import TradeCostAnalyzer
    analyzer = TradeCostAnalyzer(trades_df)
    results = analyzer.analyze()
    analyzer.generate_report("reports/trade_costs.html")
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
DEFAULT_OUTPUT = ROOT / "reports" / "trade_costs.html"


# ── Cost config defaults ────────────────────────────────────────────────

DEFAULT_COMMISSION_PER_CONTRACT = 0.65
DEFAULT_EXCHANGE_FEE = 0.30
DEFAULT_REGULATORY_FEE = 0.03
DEFAULT_TAX_RATE = 0.0             # varies by jurisdiction
DEFAULT_DAILY_VOL = 0.015          # annualized σ proxy for options
DEFAULT_AVG_DAILY_VOLUME = 5000    # contracts
DEFAULT_RISK_AVERSION = 1e-6       # Almgren-Chriss λ


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class ExplicitCosts:
    """Explicit, directly observable trade costs."""
    commissions: float
    exchange_fees: float
    regulatory_fees: float
    taxes: float
    total: float


@dataclass
class ImplicitCosts:
    """Implicit costs from market microstructure."""
    spread_cost: float        # half-spread × contracts × multiplier
    slippage: float           # execution price vs mid
    market_impact: float      # Almgren-Chriss temporary + permanent
    total: float


@dataclass
class OpportunityCosts:
    """Costs from delay or inaction."""
    delay_cost: float         # price drift during delay
    missed_trade_cost: float  # expected P&L of trades not taken
    total: float


@dataclass
class TradeCost:
    """Complete cost breakdown for a single trade."""
    trade_id: str
    strategy: str
    asset: str
    contracts: int
    explicit: ExplicitCosts
    implicit: ImplicitCosts
    opportunity: OpportunityCosts
    total_cost: float
    cost_as_pct_of_premium: float
    date: str


@dataclass
class StrategyAttribution:
    """Cost attribution for one strategy."""
    strategy: str
    n_trades: int
    total_cost: float
    avg_cost_per_trade: float
    explicit_pct: float
    implicit_pct: float
    opportunity_pct: float
    cost_as_pct_of_pnl: float


@dataclass
class TimeBucket:
    """Cost aggregation for a time period."""
    period: str
    n_trades: int
    total_cost: float
    avg_cost: float
    explicit: float
    implicit: float
    opportunity: float


@dataclass
class CostForecast:
    """Forecasted cost for a hypothetical trade."""
    contracts: int
    explicit: float
    implicit: float
    opportunity: float
    total: float
    optimal_contracts: int
    optimal_cost: float


@dataclass
class OptimizationRec:
    """Trade optimization recommendation."""
    category: str          # "size", "timing", "structure"
    description: str
    estimated_savings: float
    confidence: str        # "high", "medium", "low"


# ── Almgren-Chriss market impact model ──────────────────────────────────


def almgren_chriss_impact(
    contracts: int,
    daily_volume: float,
    daily_vol: float,
    price: float,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    n_periods: int = 1,
) -> Tuple[float, float]:
    """Compute temporary and permanent market impact.

    Returns (temporary_impact, permanent_impact) in dollar terms.
    Simplified single-period Almgren-Chriss.
    """
    if daily_volume <= 0 or price <= 0:
        return 0.0, 0.0

    participation = contracts / max(daily_volume, 1)
    sigma = daily_vol * price

    # Temporary impact: η × (n/V) × σ
    eta = 0.142                # empirical temporary impact coefficient
    temp = eta * sigma * math.sqrt(participation)

    # Permanent impact: γ × (n/V) × σ
    gamma = 0.314              # empirical permanent impact coefficient
    perm = gamma * sigma * participation

    temp_dollar = temp * contracts * 100  # 100 shares per contract
    perm_dollar = perm * contracts * 100

    return temp_dollar, perm_dollar


# ── Analyzer ────────────────────────────────────────────────────────────


class TradeCostAnalyzer:
    """Comprehensive trade cost analysis."""

    def __init__(
        self,
        trades: pd.DataFrame,
        commission: float = DEFAULT_COMMISSION_PER_CONTRACT,
        exchange_fee: float = DEFAULT_EXCHANGE_FEE,
        regulatory_fee: float = DEFAULT_REGULATORY_FEE,
        tax_rate: float = DEFAULT_TAX_RATE,
        avg_daily_volume: float = DEFAULT_AVG_DAILY_VOLUME,
        daily_vol: float = DEFAULT_DAILY_VOL,
        risk_aversion: float = DEFAULT_RISK_AVERSION,
    ) -> None:
        self.trades = trades.copy()
        self.commission = commission
        self.exchange_fee = exchange_fee
        self.regulatory_fee = regulatory_fee
        self.tax_rate = tax_rate
        self.avg_daily_volume = avg_daily_volume
        self.daily_vol = daily_vol
        self.risk_aversion = risk_aversion

        # Ensure required columns have defaults
        if "strategy" not in self.trades.columns:
            self.trades["strategy"] = "default"
        if "asset" not in self.trades.columns:
            self.trades["asset"] = "SPY"
        if "contracts" not in self.trades.columns:
            self.trades["contracts"] = 1
        if "spread_width" not in self.trades.columns:
            self.trades["spread_width"] = 5.0
        if "net_credit" not in self.trades.columns:
            self.trades["net_credit"] = 1.0
        if "bid_ask_spread" not in self.trades.columns:
            self.trades["bid_ask_spread"] = 0.05
        if "slippage" not in self.trades.columns:
            self.trades["slippage"] = 0.02
        if "delay_minutes" not in self.trades.columns:
            self.trades["delay_minutes"] = 0.0
        if "underlying_price" not in self.trades.columns:
            self.trades["underlying_price"] = 430.0

        # Results
        self.trade_costs: List[TradeCost] = []
        self.strategy_attribution: List[StrategyAttribution] = []
        self.time_buckets: List[TimeBucket] = []
        self.optimizations: List[OptimizationRec] = []

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> "TradeCostAnalyzer":
        """Load trades from CSV."""
        df = pd.read_csv(path, parse_dates=True)
        return cls(df, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """Run full cost analysis."""
        self.trade_costs = self._compute_all_costs()
        self.strategy_attribution = self._attribute_by_strategy()
        self.time_buckets = self._attribute_by_time()
        self.optimizations = self._generate_recommendations()
        return {
            "trade_costs": self.trade_costs,
            "strategy_attribution": self.strategy_attribution,
            "time_buckets": self.time_buckets,
            "optimizations": self.optimizations,
        }

    def forecast(
        self,
        contracts: int,
        spread_width: float = 5.0,
        net_credit: float = 1.0,
        bid_ask: float = 0.05,
        delay_minutes: float = 0.0,
        underlying_price: float = 430.0,
    ) -> CostForecast:
        """Forecast costs for a hypothetical trade."""
        explicit = self._explicit_costs(contracts, net_credit)
        implicit = self._implicit_costs(
            contracts, bid_ask, 0.02, underlying_price,
        )
        opportunity = self._opportunity_costs(
            contracts, delay_minutes, underlying_price, net_credit,
        )
        total = explicit.total + implicit.total + opportunity.total

        # Find optimal size (minimize cost per contract)
        best_size = contracts
        best_cpc = total / max(contracts, 1)
        for c in range(max(1, contracts // 2), contracts * 2 + 1):
            e = self._explicit_costs(c, net_credit)
            i = self._implicit_costs(c, bid_ask, 0.02, underlying_price)
            o = self._opportunity_costs(c, delay_minutes, underlying_price, net_credit)
            cpc = (e.total + i.total + o.total) / c
            if cpc < best_cpc:
                best_cpc = cpc
                best_size = c

        opt_e = self._explicit_costs(best_size, net_credit)
        opt_i = self._implicit_costs(best_size, bid_ask, 0.02, underlying_price)
        opt_o = self._opportunity_costs(best_size, delay_minutes, underlying_price, net_credit)

        return CostForecast(
            contracts=contracts,
            explicit=explicit.total,
            implicit=implicit.total,
            opportunity=opportunity.total,
            total=total,
            optimal_contracts=best_size,
            optimal_cost=opt_e.total + opt_i.total + opt_o.total,
        )

    # ── Cost computation ────────────────────────────────────────────────

    def _compute_all_costs(self) -> List[TradeCost]:
        results: List[TradeCost] = []
        for i, row in self.trades.iterrows():
            contracts = int(row["contracts"])
            credit = float(row["net_credit"])
            bid_ask = float(row.get("bid_ask_spread", 0.05))
            slip = float(row.get("slippage", 0.02))
            delay = float(row.get("delay_minutes", 0))
            price = float(row.get("underlying_price", 430))

            explicit = self._explicit_costs(contracts, credit)
            implicit = self._implicit_costs(contracts, bid_ask, slip, price)
            opportunity = self._opportunity_costs(contracts, delay, price, credit)
            total = explicit.total + implicit.total + opportunity.total

            premium = credit * contracts * 100
            cost_pct = total / premium if premium > 0 else 0.0

            date_val = row.get("entry_date", row.get("date", ""))
            results.append(TradeCost(
                trade_id=str(row.get("trade_id", i)),
                strategy=str(row["strategy"]),
                asset=str(row["asset"]),
                contracts=contracts,
                explicit=explicit,
                implicit=implicit,
                opportunity=opportunity,
                total_cost=total,
                cost_as_pct_of_premium=cost_pct,
                date=str(date_val),
            ))
        return results

    def _explicit_costs(self, contracts: int, net_credit: float) -> ExplicitCosts:
        """Compute explicit costs for a trade."""
        # 2 legs × contracts (open + close = 4 transactions)
        n_transactions = contracts * 4
        comm = self.commission * n_transactions
        exch = self.exchange_fee * n_transactions
        reg = self.regulatory_fee * n_transactions
        premium = net_credit * contracts * 100
        taxes = premium * self.tax_rate
        return ExplicitCosts(
            commissions=comm, exchange_fees=exch, regulatory_fees=reg,
            taxes=taxes, total=comm + exch + reg + taxes,
        )

    def _implicit_costs(
        self, contracts: int, bid_ask: float, slippage: float,
        underlying_price: float,
    ) -> ImplicitCosts:
        """Compute implicit costs: spread, slippage, market impact."""
        multiplier = 100.0
        # Spread cost: half-spread per leg × 2 legs × contracts
        spread_cost = (bid_ask / 2) * 2 * contracts * multiplier
        # Slippage
        slip_cost = slippage * contracts * multiplier

        # Market impact (Almgren-Chriss)
        temp, perm = almgren_chriss_impact(
            contracts, self.avg_daily_volume, self.daily_vol,
            underlying_price, self.risk_aversion,
        )
        impact = temp + perm

        return ImplicitCosts(
            spread_cost=spread_cost, slippage=slip_cost,
            market_impact=impact, total=spread_cost + slip_cost + impact,
        )

    def _opportunity_costs(
        self, contracts: int, delay_minutes: float,
        underlying_price: float, net_credit: float,
    ) -> OpportunityCosts:
        """Compute opportunity costs from execution delay."""
        multiplier = 100.0
        # Delay cost: expected price drift during delay
        # σ_minute ≈ σ_daily / √(390 minutes)
        sigma_min = self.daily_vol * underlying_price / math.sqrt(390)
        delay_cost = sigma_min * math.sqrt(max(delay_minutes, 0)) * contracts * multiplier * 0.01

        # Missed trade cost: assume small percentage of trades are missed
        # due to stale signals
        missed_cost = 0.0
        if delay_minutes > 30:
            # P(miss) scales with delay
            p_miss = min(delay_minutes / 480.0, 0.5)
            expected_pnl = net_credit * contracts * multiplier * 0.3  # 30% of premium as expected P&L
            missed_cost = p_miss * expected_pnl

        return OpportunityCosts(
            delay_cost=delay_cost, missed_trade_cost=missed_cost,
            total=delay_cost + missed_cost,
        )

    # ── Attribution ─────────────────────────────────────────────────────

    def _attribute_by_strategy(self) -> List[StrategyAttribution]:
        """Aggregate costs by strategy."""
        if not self.trade_costs:
            return []
        groups: Dict[str, List[TradeCost]] = {}
        for tc in self.trade_costs:
            groups.setdefault(tc.strategy, []).append(tc)

        results: List[StrategyAttribution] = []
        for strat, costs in sorted(groups.items()):
            total = sum(tc.total_cost for tc in costs)
            explicit_sum = sum(tc.explicit.total for tc in costs)
            implicit_sum = sum(tc.implicit.total for tc in costs)
            opp_sum = sum(tc.opportunity.total for tc in costs)

            pnl = 0.0
            for tc in costs:
                row_mask = self.trades["strategy"] == strat
                if "pnl" in self.trades.columns:
                    pnl = float(self.trades.loc[row_mask, "pnl"].sum())

            results.append(StrategyAttribution(
                strategy=strat, n_trades=len(costs),
                total_cost=total,
                avg_cost_per_trade=total / len(costs),
                explicit_pct=explicit_sum / total if total > 0 else 0,
                implicit_pct=implicit_sum / total if total > 0 else 0,
                opportunity_pct=opp_sum / total if total > 0 else 0,
                cost_as_pct_of_pnl=total / abs(pnl) if abs(pnl) > 0 else 0,
            ))
        return sorted(results, key=lambda a: -a.total_cost)

    def _attribute_by_time(self) -> List[TimeBucket]:
        """Aggregate costs by time period (monthly)."""
        if not self.trade_costs:
            return []

        buckets: Dict[str, List[TradeCost]] = {}
        for tc in self.trade_costs:
            # Try to parse date for monthly bucketing
            try:
                dt = pd.Timestamp(tc.date)
                key = dt.strftime("%Y-%m")
            except (ValueError, TypeError):
                key = "unknown"
            buckets.setdefault(key, []).append(tc)

        results: List[TimeBucket] = []
        for period, costs in sorted(buckets.items()):
            total = sum(tc.total_cost for tc in costs)
            results.append(TimeBucket(
                period=period, n_trades=len(costs),
                total_cost=total,
                avg_cost=total / len(costs),
                explicit=sum(tc.explicit.total for tc in costs),
                implicit=sum(tc.implicit.total for tc in costs),
                opportunity=sum(tc.opportunity.total for tc in costs),
            ))
        return results

    # ── Optimization recommendations ────────────────────────────────────

    def _generate_recommendations(self) -> List[OptimizationRec]:
        """Generate actionable cost reduction recommendations."""
        recs: List[OptimizationRec] = []
        if not self.trade_costs:
            return recs

        total_explicit = sum(tc.explicit.total for tc in self.trade_costs)
        total_implicit = sum(tc.implicit.total for tc in self.trade_costs)
        total_opp = sum(tc.opportunity.total for tc in self.trade_costs)
        total_all = total_explicit + total_implicit + total_opp

        # Size optimization
        avg_contracts = np.mean([tc.contracts for tc in self.trade_costs])
        if avg_contracts > 3:
            savings = total_implicit * 0.15
            recs.append(OptimizationRec(
                category="size",
                description=f"Reduce average position size from {avg_contracts:.0f} to {avg_contracts * 0.7:.0f} contracts to lower market impact",
                estimated_savings=savings,
                confidence="medium",
            ))

        # Spread cost dominance
        spread_total = sum(tc.implicit.spread_cost for tc in self.trade_costs)
        if total_all > 0 and spread_total / total_all > 0.3:
            recs.append(OptimizationRec(
                category="structure",
                description="Use limit orders at mid-price to reduce spread costs",
                estimated_savings=spread_total * 0.4,
                confidence="high",
            ))

        # Delay cost
        if total_opp > total_all * 0.1:
            recs.append(OptimizationRec(
                category="timing",
                description="Reduce execution delay — automate order routing to cut opportunity costs",
                estimated_savings=total_opp * 0.5,
                confidence="medium",
            ))

        # Slippage reduction
        slip_total = sum(tc.implicit.slippage for tc in self.trade_costs)
        if slip_total > total_all * 0.15:
            recs.append(OptimizationRec(
                category="timing",
                description="Avoid trading in first/last 15 minutes to reduce slippage",
                estimated_savings=slip_total * 0.3,
                confidence="low",
            ))

        return sorted(recs, key=lambda r: -r.estimated_savings)

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate HTML report."""
        if not self.trade_costs:
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
        charts["waterfall"] = self._chart_waterfall()
        charts["strategy_comparison"] = self._chart_strategy_comparison()
        charts["cost_trend"] = self._chart_cost_trend()
        charts["cost_breakdown_pie"] = self._chart_breakdown_pie()
        return charts

    def _chart_waterfall(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.trade_costs:
            return ""

        total_explicit = sum(tc.explicit.total for tc in self.trade_costs)
        total_spread = sum(tc.implicit.spread_cost for tc in self.trade_costs)
        total_slip = sum(tc.implicit.slippage for tc in self.trade_costs)
        total_impact = sum(tc.implicit.market_impact for tc in self.trade_costs)
        total_delay = sum(tc.opportunity.delay_cost for tc in self.trade_costs)
        total_missed = sum(tc.opportunity.missed_trade_cost for tc in self.trade_costs)

        labels = ["Commissions\n& Fees", "Spread", "Slippage", "Market\nImpact", "Delay", "Missed\nTrades", "TOTAL"]
        values = [total_explicit, total_spread, total_slip, total_impact, total_delay, total_missed]
        grand_total = sum(values)
        values.append(grand_total)

        # Waterfall positions
        bottoms = [0.0]
        for v in values[:-2]:
            bottoms.append(bottoms[-1] + v)
        bottoms.append(0.0)  # total bar starts at 0

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#3b82f6", "#f59e0b", "#f59e0b", "#dc2626", "#8b5cf6", "#8b5cf6", "#1e293b"]
        for i, (lbl, val, bot) in enumerate(zip(labels, values, bottoms)):
            ax.bar(i, val, bottom=bot, color=colors[i], alpha=0.85, edgecolor="white")
            ax.text(i, bot + val + grand_total * 0.02, f"${val:,.0f}",
                    ha="center", fontsize=7, fontweight="bold")
        # Connector lines
        for i in range(len(values) - 2):
            top = bottoms[i] + values[i]
            ax.plot([i + 0.4, i + 0.6], [top, top], color="#94a3b8", lw=0.8)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Cost ($)")
        ax.set_title("Trade Cost Waterfall", fontsize=11)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_strategy_comparison(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.strategy_attribution:
            return ""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        names = [a.strategy for a in self.strategy_attribution]
        totals = [a.total_cost for a in self.strategy_attribution]
        avg_costs = [a.avg_cost_per_trade for a in self.strategy_attribution]

        ax1.barh(names, totals, color="#3b82f6", alpha=0.85)
        ax1.set_xlabel("Total Cost ($)")
        ax1.set_title("Total Cost by Strategy", fontsize=10)
        ax1.grid(True, axis="x", alpha=0.3)

        ax2.barh(names, avg_costs, color="#f59e0b", alpha=0.85)
        ax2.set_xlabel("Avg Cost per Trade ($)")
        ax2.set_title("Average Cost by Strategy", fontsize=10)
        ax2.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_cost_trend(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.time_buckets:
            return ""
        fig, ax = plt.subplots(figsize=(9, 4))
        periods = [b.period for b in self.time_buckets]
        xs = range(len(periods))

        explicit = [b.explicit for b in self.time_buckets]
        implicit = [b.implicit for b in self.time_buckets]
        opp = [b.opportunity for b in self.time_buckets]

        ax.bar(xs, explicit, label="Explicit", color="#3b82f6", alpha=0.85)
        ax.bar(xs, implicit, bottom=explicit, label="Implicit", color="#f59e0b", alpha=0.85)
        bottoms = [e + i for e, i in zip(explicit, implicit)]
        ax.bar(xs, opp, bottom=bottoms, label="Opportunity", color="#8b5cf6", alpha=0.85)

        ax.set_xticks(xs)
        ax.set_xticklabels(periods, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Cost ($)")
        ax.set_title("Cost Trend Over Time", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_breakdown_pie(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.trade_costs:
            return ""
        total_comm = sum(tc.explicit.commissions for tc in self.trade_costs)
        total_exch = sum(tc.explicit.exchange_fees for tc in self.trade_costs)
        total_spread = sum(tc.implicit.spread_cost for tc in self.trade_costs)
        total_slip = sum(tc.implicit.slippage for tc in self.trade_costs)
        total_impact = sum(tc.implicit.market_impact for tc in self.trade_costs)
        total_delay = sum(tc.opportunity.delay_cost for tc in self.trade_costs)
        total_missed = sum(tc.opportunity.missed_trade_cost for tc in self.trade_costs)

        labels = ["Commissions", "Exchange Fees", "Spread", "Slippage", "Market Impact", "Delay", "Missed"]
        sizes = [total_comm, total_exch, total_spread, total_slip, total_impact, total_delay, total_missed]
        # Filter zeros
        nonzero = [(l, s) for l, s in zip(labels, sizes) if s > 0]
        if not nonzero:
            return ""
        labels, sizes = zip(*nonzero)
        colors = ["#3b82f6", "#60a5fa", "#f59e0b", "#fbbf24", "#dc2626", "#8b5cf6", "#a78bfa"]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct="%1.0f%%",
               startangle=90, pctdistance=0.8, textprops={"fontsize": 8})
        ax.set_title("Cost Category Breakdown", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        total_cost = sum(tc.total_cost for tc in self.trade_costs)
        total_explicit = sum(tc.explicit.total for tc in self.trade_costs)
        total_implicit = sum(tc.implicit.total for tc in self.trade_costs)
        total_opp = sum(tc.opportunity.total for tc in self.trade_costs)
        n_trades = len(self.trade_costs)
        avg_cost = total_cost / n_trades if n_trades > 0 else 0

        # Strategy attribution table
        strat_rows = ""
        for a in self.strategy_attribution:
            strat_rows += (
                f'<tr><td>{a.strategy}</td><td>{a.n_trades}</td>'
                f'<td>${a.total_cost:,.2f}</td><td>${a.avg_cost_per_trade:,.2f}</td>'
                f'<td>{a.explicit_pct:.0%}</td><td>{a.implicit_pct:.0%}</td>'
                f'<td>{a.opportunity_pct:.0%}</td></tr>\n'
            )

        # Time trend table
        time_rows = ""
        for b in self.time_buckets:
            time_rows += (
                f'<tr><td>{b.period}</td><td>{b.n_trades}</td>'
                f'<td>${b.total_cost:,.2f}</td><td>${b.avg_cost:,.2f}</td>'
                f'<td>${b.explicit:,.2f}</td><td>${b.implicit:,.2f}</td>'
                f'<td>${b.opportunity:,.2f}</td></tr>\n'
            )

        # Recommendations table
        rec_rows = ""
        for r in self.optimizations:
            conf_cls = {"high": "good", "medium": "warn", "low": "bad"}.get(r.confidence, "")
            rec_rows += (
                f'<tr><td>{r.category.upper()}</td><td>{r.description}</td>'
                f'<td class="good">${r.estimated_savings:,.2f}</td>'
                f'<td class="{conf_cls}">{r.confidence}</td></tr>\n'
            )
        if not rec_rows:
            rec_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">No recommendations</td></tr>'

        # Top 20 costliest trades
        top_trades = sorted(self.trade_costs, key=lambda t: -t.total_cost)[:20]
        trade_rows = ""
        for tc in top_trades:
            trade_rows += (
                f'<tr><td>{tc.strategy}</td><td>{tc.asset}</td>'
                f'<td>{tc.contracts}</td>'
                f'<td>${tc.explicit.total:,.2f}</td>'
                f'<td>${tc.implicit.total:,.2f}</td>'
                f'<td>${tc.opportunity.total:,.2f}</td>'
                f'<td>${tc.total_cost:,.2f}</td>'
                f'<td>{tc.cost_as_pct_of_premium:.1%}</td></tr>\n'
            )

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trade Cost Analysis</title>
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

<h1>Trade Cost Analysis</h1>
<div class="meta">{n_trades} trades &middot; {len(self.strategy_attribution)} strategies &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value bad">${total_cost:,.0f}</div><div class="label">Total Cost</div></div>
  <div class="kpi"><div class="value">${avg_cost:,.2f}</div><div class="label">Avg Cost / Trade</div></div>
  <div class="kpi"><div class="value">${total_explicit:,.0f}</div><div class="label">Explicit</div></div>
  <div class="kpi"><div class="value">${total_implicit:,.0f}</div><div class="label">Implicit</div></div>
  <div class="kpi"><div class="value">${total_opp:,.0f}</div><div class="label">Opportunity</div></div>
</div>

<h2>1. Cost Breakdown Waterfall</h2>
{_img("waterfall")}
{_img("cost_breakdown_pie")}

<h2>2. Strategy Comparison</h2>
{_img("strategy_comparison")}
<table>
<thead><tr><th>Strategy</th><th>Trades</th><th>Total Cost</th><th>Avg/Trade</th><th>Explicit %</th><th>Implicit %</th><th>Opportunity %</th></tr></thead>
<tbody>{strat_rows}</tbody>
</table>

<h2>3. Cost Trends</h2>
{_img("cost_trend")}
<table>
<thead><tr><th>Period</th><th>Trades</th><th>Total</th><th>Avg</th><th>Explicit</th><th>Implicit</th><th>Opportunity</th></tr></thead>
<tbody>{time_rows}</tbody>
</table>

<h2>4. Costliest Trades</h2>
<table>
<thead><tr><th>Strategy</th><th>Asset</th><th>Contracts</th><th>Explicit</th><th>Implicit</th><th>Opportunity</th><th>Total</th><th>% of Premium</th></tr></thead>
<tbody>{trade_rows}</tbody>
</table>

<h2>5. Optimization Recommendations</h2>
<table>
<thead><tr><th>Category</th><th>Recommendation</th><th>Est. Savings</th><th>Confidence</th></tr></thead>
<tbody>{rec_rows}</tbody>
</table>

<footer>Generated by <code>compass/trade_cost_analyzer.py</code></footer>
</body></html>"""
        return html
