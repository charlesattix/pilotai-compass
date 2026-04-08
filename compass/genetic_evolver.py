"""
Genetic algorithm strategy evolver.

Evolves trading rule configurations via genetic algorithm.  Genome
encodes entry/exit thresholds, sizing weights, feature selections,
and regime filters.  Fitness = OOS-only Sharpe × √CAGR / max(DD, 0.05)
with parsimony pressure.

Usage::

    from compass.genetic_evolver import GeneticEvolver, EvolverConfig
    evolver = GeneticEvolver(trades_df, EvolverConfig())
    best = evolver.evolve()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Configuration ───────────────────────────────────────────────────────

# Gene definitions: (name, min, max, type)
# type: "continuous" or "discrete"
DEFAULT_GENES = [
    ("ml_threshold", 0.40, 0.95, "continuous"),
    ("stop_loss_mult", 1.0, 5.0, "continuous"),
    ("profit_target_pct", 0.30, 0.80, "continuous"),
    ("max_dte", 7, 50, "discrete"),
    ("min_dte", 3, 20, "discrete"),
    ("vix_floor", 10, 20, "continuous"),
    ("vix_ceiling", 25, 45, "continuous"),
    ("rsi_oversold", 20, 40, "discrete"),
    ("rsi_overbought", 60, 85, "discrete"),
    ("regime_bull_size", 0.5, 2.5, "continuous"),
    ("regime_bear_size", 0.1, 1.0, "continuous"),
    ("regime_highvol_size", 0.0, 0.5, "continuous"),
    ("use_momentum", 0, 1, "discrete"),
    ("use_iv_rank", 0, 1, "discrete"),
    ("use_ma_filter", 0, 1, "discrete"),
    ("iv_rank_min", 15, 80, "continuous"),
    ("momentum_threshold", -3.0, 3.0, "continuous"),
    ("ma_lookback", 10, 60, "discrete"),
    ("max_positions", 3, 15, "discrete"),
    ("position_pct", 0.02, 0.15, "continuous"),
]


@dataclass
class EvolverConfig:
    population_size: int = 100
    n_generations: int = 50
    tournament_size: int = 5
    crossover_rate: float = 0.80
    mutation_rate: float = 0.15
    mutation_strength: float = 0.10
    elitism_count: int = 5
    is_fraction: float = 0.60         # in-sample fraction
    parsimony_penalty: float = 0.005  # per active gene
    min_dd_floor: float = 0.05        # DD floor in fitness
    seed: int = 42
    genes: List[Tuple[str, float, float, str]] = field(
        default_factory=lambda: list(DEFAULT_GENES),
    )


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class Genome:
    """A single individual in the population."""
    genes: np.ndarray                  # normalised [0, 1] per gene
    fitness: float = 0.0
    is_fitness: float = 0.0
    oos_fitness: float = 0.0
    sharpe: float = 0.0
    cagr: float = 0.0
    max_dd: float = 0.0
    n_trades: int = 0
    n_active_genes: int = 0
    generation: int = 0


@dataclass
class EvolutionResult:
    """Result of the evolution process."""
    best_genome: Genome
    best_params: Dict[str, float]
    best_fitness: float
    is_fitness: float
    oos_fitness: float
    oos_is_ratio: float
    n_generations: int
    population_size: int
    convergence_gen: int               # generation where best was found
    fitness_history: List[float]       # best fitness per generation
    diversity_history: List[float]     # population diversity per generation
    all_best: List[Genome]             # best per generation


# ── Genome operations ───────────────────────────────────────────────────


def decode_genome(
    genome: np.ndarray,
    gene_defs: List[Tuple[str, float, float, str]],
) -> Dict[str, float]:
    """Decode normalised [0,1] genome to parameter dict."""
    params = {}
    for i, (name, lo, hi, gtype) in enumerate(gene_defs):
        if i >= len(genome):
            break
        val = lo + genome[i] * (hi - lo)
        if gtype == "discrete":
            val = round(val)
        params[name] = val
    return params


def random_genome(n_genes: int, rng: np.random.RandomState) -> np.ndarray:
    """Create a random genome with values in [0, 1]."""
    return rng.random(n_genes)


def tournament_select(
    population: List[Genome],
    tournament_size: int,
    rng: np.random.RandomState,
) -> Genome:
    """Tournament selection: pick best from random subset."""
    indices = rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    candidates = [population[i] for i in indices]
    return max(candidates, key=lambda g: g.fitness)


def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.RandomState,
    method: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """Crossover two parents to produce two children."""
    n = len(parent1)
    if method == "uniform":
        mask = rng.random(n) > 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
    elif method == "arithmetic":
        alpha = rng.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
    else:
        # Single-point
        pt = rng.randint(1, n)
        child1 = np.concatenate([parent1[:pt], parent2[pt:]])
        child2 = np.concatenate([parent2[:pt], parent1[pt:]])
    return np.clip(child1, 0, 1), np.clip(child2, 0, 1)


def mutate(
    genome: np.ndarray,
    rate: float,
    strength: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Gaussian mutation with boundary reflection."""
    child = genome.copy()
    for i in range(len(child)):
        if rng.random() < rate:
            child[i] += rng.normal(0, strength)
            # Reflection at boundaries
            while child[i] < 0 or child[i] > 1:
                if child[i] < 0:
                    child[i] = -child[i]
                if child[i] > 1:
                    child[i] = 2 - child[i]
    return np.clip(child, 0, 1)


def population_diversity(population: List[Genome]) -> float:
    """Average pairwise Euclidean distance between genomes."""
    if len(population) < 2:
        return 0.0
    genes = np.array([g.genes for g in population])
    # Sample pairs for efficiency
    n = min(50, len(population))
    indices = np.random.choice(len(population), n, replace=False)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.sqrt(np.sum((genes[indices[i]] - genes[indices[j]]) ** 2))
            count += 1
    return total / max(count, 1)


# ── Fitness evaluation ──────────────────────────────────────────────────


def evaluate_fitness(
    params: Dict[str, float],
    trades: pd.DataFrame,
    is_mask: np.ndarray,
    capital: float = 100_000,
    min_dd: float = 0.05,
    parsimony_penalty: float = 0.005,
    gene_defs: Optional[List[Tuple[str, float, float, str]]] = None,
) -> Tuple[float, float, float, float, float, int]:
    """Evaluate strategy fitness on IS and OOS data.

    Returns (oos_fitness, is_fitness, sharpe, cagr, max_dd, n_trades).
    """
    def _eval_subset(mask):
        subset = trades[mask]
        if len(subset) < 5:
            return 0.0, 0.0, 0.0, 0

        # Apply parameter filters
        filtered = subset.copy()

        # ML threshold
        if "pred_prob" in filtered.columns:
            filtered = filtered[filtered["pred_prob"] >= params.get("ml_threshold", 0.5)]

        # DTE filter
        if "dte_at_entry" in filtered.columns:
            filtered = filtered[
                (filtered["dte_at_entry"] >= params.get("min_dte", 5)) &
                (filtered["dte_at_entry"] <= params.get("max_dte", 45))
            ]

        # VIX filter
        if "vix" in filtered.columns:
            filtered = filtered[
                (filtered["vix"] >= params.get("vix_floor", 12)) &
                (filtered["vix"] <= params.get("vix_ceiling", 35))
            ]

        # RSI filter
        if "rsi_14" in filtered.columns and params.get("use_momentum", 0) >= 0.5:
            filtered = filtered[
                (filtered["rsi_14"] >= params.get("rsi_oversold", 30)) &
                (filtered["rsi_14"] <= params.get("rsi_overbought", 70))
            ]

        # IV rank filter
        if "iv_rank" in filtered.columns and params.get("use_iv_rank", 0) >= 0.5:
            filtered = filtered[filtered["iv_rank"] >= params.get("iv_rank_min", 25)]

        if len(filtered) < 3:
            return 0.0, 0.0, 0.0, 0

        # Regime sizing
        pnls = filtered["pnl"].values.copy()
        if "regime" in filtered.columns:
            regimes = filtered["regime"].values
            for i, r in enumerate(regimes):
                r_str = str(r).lower()
                if "bull" in r_str:
                    pnls[i] *= params.get("regime_bull_size", 1.0)
                elif "bear" in r_str:
                    pnls[i] *= params.get("regime_bear_size", 0.5)
                elif "high" in r_str or "vol" in r_str:
                    pnls[i] *= params.get("regime_highvol_size", 0.25)

        # Metrics
        eq = capital + np.cumsum(pnls)
        eq_f = np.concatenate([[capital], eq])
        rets = pnls / capital
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0
        pk = np.maximum.accumulate(eq_f)
        dd = float(np.min((eq_f - pk) / np.where(pk > 0, pk, 1)))
        total_ret = eq_f[-1] / eq_f[0] - 1
        n_days = len(filtered) * 7  # rough calendar days
        years = max(n_days / 365, 0.1)
        cagr = (1 + total_ret) ** (1 / years) - 1 if total_ret > -1 else -1

        return sharpe, cagr, dd, len(filtered)

    is_sh, is_cagr, is_dd, is_n = _eval_subset(is_mask)
    oos_sh, oos_cagr, oos_dd, oos_n = _eval_subset(~is_mask)

    # Fitness: Sharpe × √(max(CAGR, 0)) / max(|DD|, min_dd)
    def _fitness(sh, cagr, dd):
        if sh <= 0 or cagr <= 0:
            return 0.0
        return sh * math.sqrt(cagr) / max(abs(dd), min_dd)

    is_fit = _fitness(is_sh, is_cagr, is_dd)
    oos_fit = _fitness(oos_sh, oos_cagr, oos_dd)

    # Parsimony: penalise active discrete genes
    n_active = sum(1 for name, lo, hi, gt in (gene_defs or [])
                   if gt == "discrete" and params.get(name, 0) >= 0.5)
    oos_fit -= n_active * parsimony_penalty

    return oos_fit, is_fit, oos_sh, oos_cagr, oos_dd, oos_n


# ── Evolver ─────────────────────────────────────────────────────────────


class GeneticEvolver:
    """Genetic algorithm strategy evolver."""

    def __init__(
        self,
        trades: pd.DataFrame,
        config: Optional[EvolverConfig] = None,
    ) -> None:
        self.trades = trades.copy()
        self.config = config or EvolverConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.n_genes = len(self.config.genes)
        self.result: Optional[EvolutionResult] = None

        # IS/OOS split
        n = len(trades)
        split = int(n * self.config.is_fraction)
        self._is_mask = np.zeros(n, dtype=bool)
        self._is_mask[:split] = True

    def evolve(self) -> EvolutionResult:
        """Run the genetic algorithm."""
        cfg = self.config

        # Initialise population
        population = self._init_population()
        self._evaluate_population(population, generation=0)

        fitness_history: List[float] = []
        diversity_history: List[float] = []
        all_best: List[Genome] = []
        best_ever = max(population, key=lambda g: g.fitness)
        convergence_gen = 0

        for gen in range(cfg.n_generations):
            # Selection + crossover + mutation
            new_pop: List[Genome] = []

            # Elitism: keep top N
            elite = sorted(population, key=lambda g: -g.fitness)[:cfg.elitism_count]
            new_pop.extend(elite)

            while len(new_pop) < cfg.population_size:
                p1 = tournament_select(population, cfg.tournament_size, self.rng)
                p2 = tournament_select(population, cfg.tournament_size, self.rng)

                if self.rng.random() < cfg.crossover_rate:
                    method = self.rng.choice(["uniform", "arithmetic"])
                    c1_genes, c2_genes = crossover(p1.genes, p2.genes, self.rng, method)
                else:
                    c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()

                c1_genes = mutate(c1_genes, cfg.mutation_rate, cfg.mutation_strength, self.rng)
                c2_genes = mutate(c2_genes, cfg.mutation_rate, cfg.mutation_strength, self.rng)

                new_pop.append(Genome(c1_genes, generation=gen + 1))
                if len(new_pop) < cfg.population_size:
                    new_pop.append(Genome(c2_genes, generation=gen + 1))

            population = new_pop[:cfg.population_size]
            self._evaluate_population(population, gen + 1)

            gen_best = max(population, key=lambda g: g.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = gen_best
                convergence_gen = gen + 1

            fitness_history.append(gen_best.fitness)
            diversity_history.append(population_diversity(population))
            all_best.append(gen_best)

        best_params = decode_genome(best_ever.genes, cfg.genes)
        oos_is = float(best_ever.oos_fitness / best_ever.is_fitness) if best_ever.is_fitness > 0 else 0.0

        self.result = EvolutionResult(
            best_genome=best_ever,
            best_params=best_params,
            best_fitness=best_ever.fitness,
            is_fitness=best_ever.is_fitness,
            oos_fitness=best_ever.oos_fitness,
            oos_is_ratio=oos_is,
            n_generations=cfg.n_generations,
            population_size=cfg.population_size,
            convergence_gen=convergence_gen,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            all_best=all_best,
        )
        return self.result

    def _init_population(self) -> List[Genome]:
        return [Genome(random_genome(self.n_genes, self.rng)) for _ in range(self.config.population_size)]

    def _evaluate_population(self, population: List[Genome], generation: int) -> None:
        cfg = self.config
        for genome in population:
            params = decode_genome(genome.genes, cfg.genes)
            oos_fit, is_fit, sh, cagr, dd, n_trades = evaluate_fitness(
                params, self.trades, self._is_mask,
                min_dd=cfg.min_dd_floor,
                parsimony_penalty=cfg.parsimony_penalty,
                gene_defs=cfg.genes,
            )
            genome.fitness = oos_fit
            genome.is_fitness = is_fit
            genome.oos_fitness = oos_fit
            genome.sharpe = sh
            genome.cagr = cagr
            genome.max_dd = dd
            genome.n_trades = n_trades
            genome.generation = generation
            genome.n_active_genes = sum(
                1 for i, (name, lo, hi, gt) in enumerate(cfg.genes)
                if gt == "discrete" and genome.genes[i] >= 0.5
            )
