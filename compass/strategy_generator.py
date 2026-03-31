"""
Automated strategy generation and screening pipeline.

Generates strategy candidates from parameter grids, backtests them,
scores by fitness (Sharpe + return + DD + robustness), filters overfit
strategies via walk-forward degradation, and evolves top performers
via genetic-algorithm-style mutation.

All methods work on pre-loaded data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class StrategyTemplate:
    """Defines a strategy candidate."""
    name: str
    params: Dict[str, Any]
    entry_signal: str = "momentum"
    exit_signal: str = "stop_loss"
    sizing_method: str = "fixed"
    max_risk_pct: float = 0.02


@dataclass
class FitnessScore:
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    robustness: float = 0.0     # 1 - walk_forward_degradation
    composite: float = 0.0


@dataclass
class StrategyResult:
    template: StrategyTemplate
    fitness: FitnessScore
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    wf_degradation: float = 0.0
    passed_filter: bool = True
    generation: int = 0


@dataclass
class GenerationSummary:
    generation: int
    n_candidates: int
    n_passed: int
    best_sharpe: float
    avg_sharpe: float


class StrategyGenerator:
    """Automated strategy generation pipeline.

    Args:
        max_wf_degradation: Max walk-forward degradation before rejection.
        fitness_weights: Weights for composite fitness score.
        n_wf_folds: Walk-forward folds.
        mutation_rate: Fraction of params mutated per generation.
    """

    def __init__(
        self,
        max_wf_degradation: float = 0.20,
        fitness_weights: Optional[Dict[str, float]] = None,
        n_wf_folds: int = 5,
        mutation_rate: float = 0.30,
    ) -> None:
        self.max_wf_degradation = max_wf_degradation
        self.fitness_weights = fitness_weights or {
            "sharpe": 0.40, "annual_return": 0.25,
            "max_drawdown": 0.20, "robustness": 0.15,
        }
        self.n_wf_folds = n_wf_folds
        self.mutation_rate = mutation_rate
        self._history: List[GenerationSummary] = []

    # ------------------------------------------------------------------
    # Parameter grid
    # ------------------------------------------------------------------

    @staticmethod
    def generate_grid(
        param_ranges: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]

    @staticmethod
    def build_templates(
        grid: List[Dict[str, Any]],
        base_name: str = "strat",
    ) -> List[StrategyTemplate]:
        """Build StrategyTemplate for each parameter set."""
        return [
            StrategyTemplate(name=f"{base_name}_{i:04d}", params=p)
            for i, p in enumerate(grid)
        ]

    # ------------------------------------------------------------------
    # Signal generation from template
    # ------------------------------------------------------------------

    @staticmethod
    def generate_signal(
        template: StrategyTemplate,
        prices: pd.Series,
    ) -> pd.Series:
        """Generate a trading signal from template params."""
        p = template.params
        lookback = p.get("lookback", 20)
        threshold = p.get("threshold", 0.0)

        if template.entry_signal == "momentum":
            mom = prices.pct_change(lookback)
            sig = mom.apply(lambda x: 1.0 if x > threshold else (-1.0 if x < -threshold else 0.0))
        elif template.entry_signal == "mean_reversion":
            ma = prices.rolling(lookback).mean()
            std = prices.rolling(lookback).std()
            z = (prices - ma) / std.replace(0, 1e-8)
            sig = z.apply(lambda x: -1.0 if x > threshold else (1.0 if x < -threshold else 0.0))
        elif template.entry_signal == "breakout":
            high = prices.rolling(lookback).max()
            low = prices.rolling(lookback).min()
            sig = pd.Series(0.0, index=prices.index)
            sig[prices >= high] = 1.0
            sig[prices <= low] = -1.0
        else:
            sig = pd.Series(0.0, index=prices.index)

        return sig.fillna(0)

    # ------------------------------------------------------------------
    # Backtest (vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def quick_backtest(
        signal: pd.Series, returns: pd.Series, cost: float = 0.001,
    ) -> pd.Series:
        """Vectorised signal backtest."""
        aligned = pd.DataFrame({"sig": signal, "ret": returns}).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)
        pos = aligned["sig"].shift(1).fillna(0)
        trades = pos.diff().abs().fillna(0)
        return pos * aligned["ret"] - trades * cost

    # ------------------------------------------------------------------
    # Fitness scoring
    # ------------------------------------------------------------------

    def compute_fitness(
        self, strat_rets: pd.Series,
        is_sharpe: float = 0.0, oos_sharpe: float = 0.0,
    ) -> FitnessScore:
        """Compute composite fitness score."""
        r = strat_rets.dropna()
        if len(r) < 10:
            return FitnessScore()

        mu = float(r.mean())
        std = float(r.std())
        sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
        total = float((1 + r).prod() - 1)
        n_years = len(r) / TRADING_DAYS
        annual = (1 + total) ** (1 / max(n_years, 0.01)) - 1
        eq = (1 + r).cumprod()
        dd = float((1 - eq / eq.expanding().max()).max())

        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe) if abs(is_sharpe) > 1e-8 else 0.0
        robustness = max(0, 1 - abs(degradation))

        fw = self.fitness_weights
        composite = (
            fw.get("sharpe", 0) * min(sharpe / 3.0, 1.0)
            + fw.get("annual_return", 0) * min(annual / 0.50, 1.0)
            + fw.get("max_drawdown", 0) * max(1 - dd / 0.30, 0)
            + fw.get("robustness", 0) * robustness
        )

        return FitnessScore(
            sharpe=sharpe, annual_return=annual,
            max_drawdown=dd, robustness=robustness,
            composite=composite,
        )

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_test(
        self, signal: pd.Series, returns: pd.Series,
    ) -> Tuple[float, float, float]:
        """Returns (is_sharpe, oos_sharpe, degradation)."""
        aligned = pd.DataFrame({"sig": signal, "ret": returns}).dropna()
        n = len(aligned)
        if n < self.n_wf_folds * 20:
            return 0.0, 0.0, 1.0

        fold = n // self.n_wf_folds
        is_sharpes, oos_sharpes = [], []

        for i in range(self.n_wf_folds - 1):
            train = aligned.iloc[:fold * (i + 1)]
            test = aligned.iloc[fold * (i + 1):fold * (i + 2)]
            if len(test) < 5:
                continue
            for chunk, bucket in [(train, is_sharpes), (test, oos_sharpes)]:
                r = self.quick_backtest(chunk["sig"], chunk["ret"])
                mu = float(r.mean())
                std = float(r.std())
                bucket.append(mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0)

        avg_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        avg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        deg = (avg_is - avg_oos) / abs(avg_is) if abs(avg_is) > 1e-8 else 0.0
        return avg_is, avg_oos, deg

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter_strategies(
        self, results: List[StrategyResult],
    ) -> List[StrategyResult]:
        """Reject overfitting strategies."""
        for r in results:
            r.passed_filter = abs(r.wf_degradation) <= self.max_wf_degradation
        return [r for r in results if r.passed_filter]

    # ------------------------------------------------------------------
    # Genetic mutation
    # ------------------------------------------------------------------

    def mutate(
        self,
        template: StrategyTemplate,
        param_ranges: Dict[str, List[Any]],
        seed: int = 42,
    ) -> StrategyTemplate:
        """Mutate a strategy template's parameters."""
        rng = np.random.default_rng(seed)
        new_params = dict(template.params)
        for key, values in param_ranges.items():
            if rng.random() < self.mutation_rate and key in new_params:
                new_params[key] = rng.choice(values)
        return StrategyTemplate(
            name=f"{template.name}_mut",
            params=new_params,
            entry_signal=template.entry_signal,
            exit_signal=template.exit_signal,
        )

    def evolve(
        self,
        top_n: List[StrategyResult],
        param_ranges: Dict[str, List[Any]],
        n_offspring: int = 10,
        seed: int = 42,
    ) -> List[StrategyTemplate]:
        """Generate offspring from top performers."""
        offspring: List[StrategyTemplate] = []
        for i, result in enumerate(top_n):
            for j in range(n_offspring // max(len(top_n), 1)):
                offspring.append(self.mutate(
                    result.template, param_ranges, seed=seed + i * 100 + j))
        return offspring[:n_offspring]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.Series,
        returns: pd.Series,
        param_ranges: Dict[str, List[Any]],
        entry_signal: str = "momentum",
        n_generations: int = 1,
        top_n: int = 5,
    ) -> Tuple[List[StrategyResult], List[GenerationSummary]]:
        """Run full generation + screening pipeline."""
        grid = self.generate_grid(param_ranges)
        templates = self.build_templates(grid)
        for t in templates:
            t.entry_signal = entry_signal

        all_results: List[StrategyResult] = []

        for gen in range(n_generations):
            gen_results: List[StrategyResult] = []
            for tmpl in templates:
                sig = self.generate_signal(tmpl, prices)
                strat_ret = self.quick_backtest(sig, returns)
                is_sh, oos_sh, deg = self.walk_forward_test(sig, returns)
                fitness = self.compute_fitness(strat_ret, is_sh, oos_sh)
                gen_results.append(StrategyResult(
                    template=tmpl, fitness=fitness,
                    is_sharpe=is_sh, oos_sharpe=oos_sh,
                    wf_degradation=deg, generation=gen,
                ))

            passed = self.filter_strategies(gen_results)
            passed.sort(key=lambda r: r.fitness.composite, reverse=True)
            all_results.extend(passed)

            best_sh = passed[0].fitness.sharpe if passed else 0.0
            avg_sh = float(np.mean([r.fitness.sharpe for r in passed])) if passed else 0.0
            self._history.append(GenerationSummary(
                gen, len(gen_results), len(passed), best_sh, avg_sh))

            # Evolve for next generation
            if gen < n_generations - 1 and passed:
                templates = self.evolve(passed[:top_n], param_ranges)

        all_results.sort(key=lambda r: r.fitness.composite, reverse=True)
        return all_results[:top_n * n_generations], self._history

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: List[StrategyResult],
        summaries: Optional[List[GenerationSummary]] = None,
        output_path: str = "reports/strategy_generator.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            f"<tr><td style='text-align:left'>{r.template.name}</td>"
            f"<td>{r.fitness.sharpe:.2f}</td><td>{r.fitness.annual_return:.2%}</td>"
            f"<td>{r.fitness.max_drawdown:.2%}</td><td>{r.fitness.robustness:.2f}</td>"
            f"<td>{r.fitness.composite:.3f}</td><td>{r.wf_degradation:.1%}</td>"
            f"<td>{r.generation}</td></tr>"
            for r in results[:20]
        ]
        gen_rows = []
        if summaries:
            gen_rows = [
                f"<tr><td>{s.generation}</td><td>{s.n_candidates}</td>"
                f"<td>{s.n_passed}</td><td>{s.best_sharpe:.2f}</td>"
                f"<td>{s.avg_sharpe:.2f}</td></tr>"
                for s in summaries
            ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Strategy Generator</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
</style></head><body>
<h1>Strategy Generator Report</h1>
<div class="summary"><p>Top {len(results)} strategies | Max WF degradation: {self.max_wf_degradation:.0%}</p></div>
<h2>Leaderboard</h2>
<table><tr><th style='text-align:left'>Name</th><th>Sharpe</th><th>Return</th>
<th>Max DD</th><th>Robustness</th><th>Fitness</th><th>WF Deg.</th><th>Gen</th></tr>
{''.join(rows)}</table>
{'<h2>Generation History</h2><table><tr><th>Gen</th><th>Candidates</th><th>Passed</th><th>Best Sharpe</th><th>Avg Sharpe</th></tr>' + ''.join(gen_rows) + '</table>' if gen_rows else ''}
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return str(path)
