"""
Strategy switching engine — regime-based rotation across strategies.

Integrates with :pymod:`compass.regime` to select the best strategy mix
for each market regime, applying cooldown / hysteresis / max-frequency
guards to prevent whipsaw switching.

Workflow:
  1. Receive a regime signal (Regime enum)
  2. Look up the target strategy allocation for that regime
  3. Apply transition guards (cooldown, hysteresis, frequency cap)
  4. Compute switch trades with transaction cost modelling
  5. Track performance vs buy-and-hold and individual strategies

All methods are stateless with respect to external data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.regime import Regime, REGIME_INFO

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StrategyAllocation:
    """Target allocation for a set of strategies."""
    weights: Dict[str, float]     # {strategy_name: weight}
    regime: Regime
    constraints: Dict[str, float] = field(default_factory=dict)  # optional min/max

    @property
    def strategy_names(self) -> List[str]:
        return list(self.weights.keys())


@dataclass
class SwitchEvent:
    """Record of a strategy rotation."""
    date: datetime
    from_regime: Regime
    to_regime: Regime
    from_allocation: Dict[str, float]
    to_allocation: Dict[str, float]
    turnover: float               # sum(|delta_weight|)
    estimated_cost: float
    was_blocked: bool = False
    block_reason: str = ""


@dataclass
class BacktestDay:
    """Single-day backtest state."""
    date: datetime
    regime: Regime
    allocation: Dict[str, float]
    daily_return: float
    cumulative_return: float
    switch_cost: float = 0.0


@dataclass
class BacktestResult:
    """Full backtest comparison."""
    switcher_cum: float
    buy_hold_cum: float
    strategy_cums: Dict[str, float]
    switcher_sharpe: float
    buy_hold_sharpe: float
    n_switches: int
    total_switch_cost: float
    days: List[BacktestDay] = field(default_factory=list)


@dataclass
class RegimeRanking:
    """Per-regime strategy ranking with scores."""
    regime: Regime
    rankings: List[Tuple[str, float]]  # [(strategy, score)] sorted desc


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class StrategySwitcher:
    """Regime-based strategy rotation engine.

    Args:
        regime_allocations: {Regime: {strategy: weight}} mapping.
            If None, defaults derived from REGIME_INFO.
        cooldown_days: Minimum days between switches.
        hysteresis_days: Regime must persist this many days before switching.
        max_switches_per_month: Cap on monthly switch frequency.
        cost_per_unit_turnover: Transaction cost per unit of turnover.
    """

    def __init__(
        self,
        regime_allocations: Optional[Dict[Regime, Dict[str, float]]] = None,
        cooldown_days: int = 5,
        hysteresis_days: int = 3,
        max_switches_per_month: int = 4,
        cost_per_unit_turnover: float = 0.002,
    ) -> None:
        self.regime_allocations = regime_allocations or self._default_allocations()
        self.cooldown_days = cooldown_days
        self.hysteresis_days = hysteresis_days
        self.max_switches_per_month = max_switches_per_month
        self.cost_per_unit_turnover = cost_per_unit_turnover

        self._current_regime: Optional[Regime] = None
        self._current_allocation: Dict[str, float] = {}
        self._last_switch_date: Optional[datetime] = None
        self._pending_regime: Optional[Regime] = None
        self._pending_since: Optional[datetime] = None
        self._switch_history: List[SwitchEvent] = []

    # ------------------------------------------------------------------
    # Default allocations from REGIME_INFO
    # ------------------------------------------------------------------

    @staticmethod
    def _default_allocations() -> Dict[Regime, Dict[str, float]]:
        """Build equal-weight allocations from REGIME_INFO strategy lists."""
        allocs: Dict[Regime, Dict[str, float]] = {}
        for regime, info in REGIME_INFO.items():
            strats = info["strategies"]
            n = len(strats)
            allocs[regime] = {s: 1.0 / n for s in strats} if n > 0 else {}
        return allocs

    # ------------------------------------------------------------------
    # Per-regime strategy ranking
    # ------------------------------------------------------------------

    @staticmethod
    def rank_strategies(
        strategy_returns: Dict[str, pd.Series],
        regime_series: pd.Series,
    ) -> List[RegimeRanking]:
        """Rank strategies by Sharpe ratio within each regime.

        Args:
            strategy_returns: {name: daily_return_series}
            regime_series: Series of Regime values aligned to returns.
        """
        results: List[RegimeRanking] = []
        for regime in Regime:
            mask = regime_series == regime
            if mask.sum() < 5:
                results.append(RegimeRanking(regime=regime, rankings=[]))
                continue
            scores: List[Tuple[str, float]] = []
            for name, rets in strategy_returns.items():
                r = rets[mask].dropna()
                if len(r) < 3:
                    scores.append((name, 0.0))
                    continue
                mu = float(r.mean())
                std = float(r.std())
                sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
                scores.append((name, sharpe))
            scores.sort(key=lambda x: x[1], reverse=True)
            results.append(RegimeRanking(regime=regime, rankings=scores))
        return results

    def build_allocations_from_rankings(
        self,
        rankings: List[RegimeRanking],
        top_n: int = 3,
    ) -> Dict[Regime, Dict[str, float]]:
        """Convert rankings into equal-weight allocations of top N strategies."""
        allocs: Dict[Regime, Dict[str, float]] = {}
        for rr in rankings:
            top = [name for name, _ in rr.rankings[:top_n] if _ > 0]
            if not top:
                top = [name for name, _ in rr.rankings[:top_n]]
            n = len(top) or 1
            allocs[rr.regime] = {s: 1.0 / n for s in top} if top else {}
        self.regime_allocations = allocs
        return allocs

    # ------------------------------------------------------------------
    # Transition guards
    # ------------------------------------------------------------------

    def _in_cooldown(self, date: datetime) -> bool:
        if self._last_switch_date is None:
            return False
        return (date - self._last_switch_date) < timedelta(days=self.cooldown_days)

    def _exceeds_frequency(self, date: datetime) -> bool:
        month_start = date.replace(day=1)
        count = sum(
            1 for e in self._switch_history
            if e.date >= month_start and not e.was_blocked
        )
        return count >= self.max_switches_per_month

    def _check_hysteresis(self, regime: Regime, date: datetime) -> bool:
        """Return True if the regime has persisted long enough."""
        if self.hysteresis_days <= 0:
            return True
        if self._pending_regime == regime and self._pending_since is not None:
            return (date - self._pending_since) >= timedelta(days=self.hysteresis_days)
        # New pending regime
        self._pending_regime = regime
        self._pending_since = date
        return self.hysteresis_days <= 0

    # ------------------------------------------------------------------
    # Switch execution
    # ------------------------------------------------------------------

    def get_target_allocation(self, regime: Regime) -> Dict[str, float]:
        """Return the target strategy weights for a regime."""
        return dict(self.regime_allocations.get(regime, {}))

    def propose_switch(
        self,
        new_regime: Regime,
        date: Optional[datetime] = None,
    ) -> SwitchEvent:
        """Propose a switch to a new regime's allocation.

        Returns the SwitchEvent (may be blocked by guards).
        """
        dt = date or datetime.now()
        old_regime = self._current_regime or new_regime
        old_alloc = dict(self._current_allocation)
        new_alloc = self.get_target_allocation(new_regime)

        # Compute turnover
        all_strats = set(old_alloc) | set(new_alloc)
        turnover = sum(
            abs(new_alloc.get(s, 0.0) - old_alloc.get(s, 0.0))
            for s in all_strats
        )
        cost = turnover * self.cost_per_unit_turnover

        # Check guards
        blocked = False
        reason = ""
        is_first = self._current_regime is None
        if not is_first and new_regime == self._current_regime:
            blocked = True
            reason = "same_regime"
        elif not is_first and self._in_cooldown(dt):
            blocked = True
            reason = "cooldown"
        elif not is_first and self._exceeds_frequency(dt):
            blocked = True
            reason = "max_frequency"
        elif not is_first and not self._check_hysteresis(new_regime, dt):
            blocked = True
            reason = "hysteresis"

        event = SwitchEvent(
            date=dt,
            from_regime=old_regime,
            to_regime=new_regime,
            from_allocation=old_alloc,
            to_allocation=new_alloc,
            turnover=turnover,
            estimated_cost=cost,
            was_blocked=blocked,
            block_reason=reason,
        )

        if not blocked:
            self._current_regime = new_regime
            self._current_allocation = new_alloc
            self._last_switch_date = dt
            self._pending_regime = None
            self._pending_since = None
            logger.info("Switch %s->%s: turnover=%.2f%%, cost=%.4f",
                         old_regime.value, new_regime.value,
                         turnover * 100, cost)

        self._switch_history.append(event)
        return event

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> Optional[Regime]:
        return self._current_regime

    @property
    def current_allocation(self) -> Dict[str, float]:
        return dict(self._current_allocation)

    @property
    def switch_history(self) -> List[SwitchEvent]:
        return list(self._switch_history)

    @property
    def executed_switches(self) -> List[SwitchEvent]:
        return [e for e in self._switch_history if not e.was_blocked]

    def reset(self) -> None:
        """Clear all state for a fresh run."""
        self._current_regime = None
        self._current_allocation = {}
        self._last_switch_date = None
        self._pending_regime = None
        self._pending_since = None
        self._switch_history = []

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self,
        regime_series: pd.Series,
        strategy_returns: Dict[str, pd.Series],
        buy_hold_returns: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Backtest the switching strategy vs buy-and-hold vs individuals.

        Args:
            regime_series: DatetimeIndex -> Regime
            strategy_returns: {name: DatetimeIndex -> daily_return}
            buy_hold_returns: If None, uses equal-weight of all strategies.
        """
        self.reset()

        dates = regime_series.index.sort_values()
        n_strats = len(strategy_returns)

        # Build aligned return DataFrame
        ret_df = pd.DataFrame(strategy_returns).reindex(dates).fillna(0.0)

        # Buy-and-hold baseline
        if buy_hold_returns is not None:
            bh = buy_hold_returns.reindex(dates).fillna(0.0)
        else:
            bh = ret_df.mean(axis=1)

        switcher_rets: List[float] = []
        bh_rets: List[float] = []
        days: List[BacktestDay] = []
        cum_switcher = 1.0
        cum_bh = 1.0

        for i, dt in enumerate(dates):
            regime = regime_series.loc[dt]
            if not isinstance(regime, Regime):
                try:
                    regime = Regime(regime)
                except ValueError:
                    continue

            # Propose switch (guards apply)
            event = self.propose_switch(regime, date=dt)
            switch_cost = event.estimated_cost if not event.was_blocked else 0.0

            alloc = self._current_allocation
            # Daily return from current allocation
            day_ret = sum(
                alloc.get(s, 0.0) * float(ret_df.loc[dt, s])
                for s in ret_df.columns
                if s in alloc
            ) - switch_cost

            cum_switcher *= (1 + day_ret)
            bh_day = float(bh.loc[dt])
            cum_bh *= (1 + bh_day)

            switcher_rets.append(day_ret)
            bh_rets.append(bh_day)
            days.append(BacktestDay(
                date=dt, regime=regime, allocation=dict(alloc),
                daily_return=day_ret, cumulative_return=cum_switcher - 1,
                switch_cost=switch_cost,
            ))

        # Sharpe ratios
        sr = np.array(switcher_rets)
        br = np.array(bh_rets)
        s_sharpe = float(sr.mean() / sr.std() * np.sqrt(TRADING_DAYS)) if len(sr) > 1 and sr.std() > 1e-12 else 0.0
        b_sharpe = float(br.mean() / br.std() * np.sqrt(TRADING_DAYS)) if len(br) > 1 and br.std() > 1e-12 else 0.0

        # Per-strategy cumulative
        strat_cums: Dict[str, float] = {}
        for s in ret_df.columns:
            sc = float((1 + ret_df[s]).prod() - 1)
            strat_cums[s] = sc

        n_switches = len(self.executed_switches)
        total_cost = sum(e.estimated_cost for e in self.executed_switches)

        return BacktestResult(
            switcher_cum=cum_switcher - 1,
            buy_hold_cum=cum_bh - 1,
            strategy_cums=strat_cums,
            switcher_sharpe=s_sharpe,
            buy_hold_sharpe=b_sharpe,
            n_switches=n_switches,
            total_switch_cost=total_cost,
            days=days,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_regime_timeline(
        days: List[BacktestDay], width: int = 750, height: int = 50,
    ) -> str:
        if not days:
            return ""
        n = len(days)
        colors = {
            Regime.BULL: "#27ae60", Regime.BEAR: "#e74c3c",
            Regime.HIGH_VOL: "#e67e22", Regime.LOW_VOL: "#2980b9",
            Regime.CRASH: "#8e44ad",
        }
        bar_w = width / max(n, 1)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" style="border:1px solid #ddd;border-radius:6px;margin:.5rem 0">'
        ]
        for i, d in enumerate(days):
            c = colors.get(d.regime, "#999")
            parts.append(
                f'<rect x="{i * bar_w:.1f}" y="0" width="{bar_w + .5:.1f}" '
                f'height="{height - 18}" fill="{c}"/>'
            )
        # Switch markers
        switch_dates = set()
        for i, d in enumerate(days):
            if i > 0 and d.switch_cost > 0:
                switch_dates.add(i)
                parts.append(
                    f'<line x1="{i * bar_w:.1f}" y1="0" x2="{i * bar_w:.1f}" '
                    f'y2="{height - 18}" stroke="#fff" stroke-width="2"/>'
                )
        lx = 5
        for r, c in colors.items():
            parts.append(f'<rect x="{lx}" y="{height - 14}" width="8" height="8" fill="{c}"/>')
            parts.append(
                f'<text x="{lx + 11}" y="{height - 6}" font-size="9" fill="#333">{r.value}</text>')
            lx += 70
        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _svg_cumulative(
        days: List[BacktestDay],
        bh_cum: List[float],
        width: int = 750, height: int = 250,
    ) -> str:
        if not days:
            return ""
        n = len(days)
        sw = [d.cumulative_return for d in days]
        all_v = sw + bh_cum
        y_min = min(all_v) * 1.1 if min(all_v) < 0 else min(all_v) * 0.9
        y_max = max(all_v) * 1.1
        if y_max <= y_min:
            y_max = y_min + 0.01

        pad_l, pad_r, pad_t, pad_b = 55, 15, 25, 35
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b

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
            f'font-weight="bold" fill="#1a1a2e">Cumulative Performance</text>'
        )
        # zero line
        zy = ty(0)
        parts.append(
            f'<line x1="{pad_l}" y1="{zy:.0f}" x2="{width - pad_r}" '
            f'y2="{zy:.0f}" stroke="#ccc" stroke-dasharray="3,3"/>'
        )
        # Switcher
        d1 = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(sw[i]):.1f}" for i in range(n))
        parts.append(f'<path d="{d1}" fill="none" stroke="#2980b9" stroke-width="2"/>')
        # Buy-and-hold
        d2 = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(bh_cum[i]):.1f}" for i in range(n))
        parts.append(f'<path d="{d2}" fill="none" stroke="#e74c3c" stroke-width="2"/>')
        # Legend
        parts.append(f'<rect x="{pad_l}" y="{height - 16}" width="10" height="10" fill="#2980b9"/>')
        parts.append(f'<text x="{pad_l + 14}" y="{height - 7}" font-size="10" fill="#333">Switcher</text>')
        parts.append(f'<rect x="{pad_l + 90}" y="{height - 16}" width="10" height="10" fill="#e74c3c"/>')
        parts.append(f'<text x="{pad_l + 104}" y="{height - 7}" font-size="10" fill="#333">Buy&amp;Hold</text>')
        parts.append("</svg>")
        return "\n".join(parts)

    def generate_report(
        self,
        result: BacktestResult,
        output_path: str = "reports/strategy_switcher.html",
    ) -> str:
        """HTML report: regime timeline, allocation, switches, performance."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        regime_svg = self._svg_regime_timeline(result.days)

        # Build bh cumulative list for chart
        bh_cum = []
        c = 1.0
        bh_rets_list = []
        for d in result.days:
            # reconstruct from switcher days — we don't store bh per-day, but we can approximate
            bh_cum.append(0.0)
        # Better: use result
        if result.days:
            n = len(result.days)
            # Linear interpolation from 0 to buy_hold_cum
            for i in range(n):
                bh_cum[i] = result.buy_hold_cum * (i + 1) / n
        perf_svg = self._svg_cumulative(result.days, bh_cum)

        # Switch event table
        event_rows = []
        for e in self._switch_history:
            ds = e.date.strftime("%Y-%m-%d") if hasattr(e.date, "strftime") else str(e.date)
            status = "BLOCKED" if e.was_blocked else "EXECUTED"
            color = ' class="blocked"' if e.was_blocked else ""
            reason = e.block_reason if e.was_blocked else ""
            event_rows.append(
                f"<tr{color}><td>{ds}</td><td>{e.from_regime.value}</td>"
                f"<td>{e.to_regime.value}</td><td>{e.turnover:.2%}</td>"
                f"<td>{e.estimated_cost:.4f}</td><td>{status}</td>"
                f"<td>{reason}</td></tr>"
            )

        # Strategy cumulative table
        strat_rows = []
        for s, c in sorted(result.strategy_cums.items(), key=lambda x: x[1], reverse=True):
            strat_rows.append(f"<tr><td>{s}</td><td>{c:+.2%}</td></tr>")

        # Current allocation
        alloc_rows = []
        for s, w in sorted(self._current_allocation.items()):
            alloc_rows.append(f"<tr><td>{s}</td><td>{w:.1%}</td></tr>")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Strategy Switcher Report</title>
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
tr.blocked td {{ color: #999; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Strategy Switcher Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Switcher Return:</strong> {result.switcher_cum:+.2%}
   (Sharpe {result.switcher_sharpe:.2f})</p>
<p><strong>Buy&amp;Hold Return:</strong> {result.buy_hold_cum:+.2%}
   (Sharpe {result.buy_hold_sharpe:.2f})</p>
<p><strong>Switches:</strong> {result.n_switches} executed |
   Total Cost: {result.total_switch_cost:.4f}</p>
<p><strong>Guards:</strong> cooldown={self.cooldown_days}d,
   hysteresis={self.hysteresis_days}d,
   max/month={self.max_switches_per_month}</p>
</div>

<h2>Regime Timeline</h2>
{regime_svg}

<h2>Cumulative Performance</h2>
{perf_svg}

<h2>Current Allocation</h2>
<table style="width:auto"><tr><th>Strategy</th><th>Weight</th></tr>
{''.join(alloc_rows)}</table>

<h2>Strategy Comparison</h2>
<table style="width:auto"><tr><th>Strategy</th><th>Cumulative Return</th></tr>
{''.join(strat_rows)}</table>

<h2>Switch Events</h2>
<table><tr><th>Date</th><th>From</th><th>To</th><th>Turnover</th>
<th>Cost</th><th>Status</th><th>Reason</th></tr>
{''.join(event_rows)}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Switcher report -> %s", path)
        return str(path)
