"""
compass/stress_test.py — Portfolio stress testing via Monte Carlo and crisis scenarios.

Provides:
  1. Monte Carlo simulation (block-bootstrap resampling of daily returns)
  2. Historical crisis scenario replay (COVID, 2022 bear, flash crash, VIX spike)
  3. Sensitivity analysis (parameter sweeps measuring Sharpe/drawdown impact)
  4. Structured results dict suitable for HTML reporting

Usage:
    from compass.stress_test import StressTester

    tester = StressTester(daily_returns, starting_capital=100_000)
    results = tester.run_all()
    # results dict has keys: monte_carlo, crisis_scenarios, sensitivity, summary
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from compass.crisis_hedge import CrisisHedgeConfig, CrisisHedgeController

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crisis path builder (must precede CRISIS_SCENARIOS)
# ---------------------------------------------------------------------------

def _build_crash_path(total_return: float, n_days: int) -> List[float]:
    """Build a daily return path that compounds to *total_return* over *n_days*.

    Adds realistic noise: returns are not uniform but follow a concave shock
    pattern where the worst days cluster early (matching empirical crash dynamics).
    """
    if n_days <= 0:
        return []
    if n_days == 1:
        return [total_return]

    # Weighted distribution: early days absorb more of the shock
    rng = np.random.RandomState(42)  # deterministic for reproducibility
    weights = np.exp(-np.linspace(0, 2, n_days))
    weights /= weights.sum()

    # Allocate log-return proportionally, then convert to simple returns
    total_log = math.log(1 + total_return)
    daily_log = total_log * weights

    # Add small noise to avoid perfectly smooth paths
    noise = rng.normal(0, abs(total_log) * 0.05, n_days)
    daily_log += noise
    # Rescale so total still matches
    daily_log *= total_log / daily_log.sum()

    daily_returns = [math.exp(lr) - 1 for lr in daily_log]
    return daily_returns


# Rebuild CRISIS_SCENARIOS now that _build_crash_path is defined
CRISIS_SCENARIOS = [
    {
        "name": "COVID Crash (Feb-Mar 2020)",
        "description": "S&P 500 fell ~34% in 23 trading days",
        "daily_shocks": _build_crash_path(-0.34, 23),
        "vix_start": 15.0,
        "vix_peak": 82.0,
    },
    {
        "name": "2022 Bear Market",
        "description": "S&P 500 fell ~25% over ~9 months (~190 trading days)",
        "daily_shocks": _build_crash_path(-0.25, 190),
        "vix_start": 17.0,
        "vix_peak": 36.0,
    },
    {
        "name": "Flash Crash (Single Day)",
        "description": "Sudden 10% drawdown in a single trading session",
        "daily_shocks": _build_crash_path(-0.10, 1),
        "vix_start": 15.0,
        "vix_peak": 65.0,
    },
    {
        "name": "VIX Spike (15 → 65)",
        "description": "VIX quadruples over 5 days; credit spreads widen sharply",
        "daily_shocks": _build_crash_path(-0.15, 5),
        "vix_start": 15.0,
        "vix_peak": 65.0,
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sharpe_ratio(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    """Annualized Sharpe ratio (risk-free rate assumed 0 for simplicity)."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * math.sqrt(annual_factor))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.15 = 15% drawdown)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))


def _cagr(equity_curve: np.ndarray, trading_days: int = 252) -> float:
    """Compound annual growth rate from an equity curve."""
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0
    total_return = equity_curve[-1] / equity_curve[0]
    years = len(equity_curve) / trading_days
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1)


def _calmar_ratio(equity_curve: np.ndarray) -> float:
    """CAGR / abs(max drawdown). Returns 0 if drawdown is zero."""
    dd = _max_drawdown(equity_curve)
    if dd == 0:
        return 0.0
    return _cagr(equity_curve) / abs(dd)


def _returns_to_equity(returns: np.ndarray, starting_capital: float) -> np.ndarray:
    """Convert a daily returns array to an equity curve."""
    cumulative = np.cumprod(1 + returns)
    return starting_capital * np.concatenate([[1.0], cumulative])


def _percentile_safe(arr: np.ndarray, q: float) -> float:
    """np.percentile wrapper that handles empty arrays."""
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, q))


# ---------------------------------------------------------------------------
# StressTester
# ---------------------------------------------------------------------------

class StressTester:
    """Portfolio stress testing engine.

    Accepts daily returns (from backtest equity curves or paper trading logs)
    and runs Monte Carlo simulations, historical crisis overlays, and parameter
    sensitivity sweeps.

    Args:
        daily_returns: Array-like of daily portfolio returns (e.g. 0.003 = +0.3%).
        starting_capital: Portfolio starting value in dollars.
        n_simulations: Number of Monte Carlo paths (default 1000).
        block_size: Block length for block-bootstrap resampling (default 5 trading days).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        daily_returns,
        starting_capital: float = 100_000,
        n_simulations: int = 1_000,
        block_size: int = 5,
        seed: int = 42,
    ):
        self.returns = np.asarray(daily_returns, dtype=np.float64)
        self.starting_capital = starting_capital
        self.n_simulations = max(n_simulations, 100)
        self.block_size = max(block_size, 1)
        self.rng = np.random.RandomState(seed)

        if len(self.returns) < 10:
            logger.warning(
                "StressTester: only %d daily returns provided; results may be unreliable",
                len(self.returns),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        param_sweep_fn: Optional[Callable] = None,
        crisis_hedge_config: Optional[CrisisHedgeConfig] = None,
    ) -> Dict[str, Any]:
        """Run the full stress testing suite.

        Args:
            param_sweep_fn: Optional callable for sensitivity analysis.
                Signature: ``fn(param_name, param_value) -> daily_returns_array``.
                If None, sensitivity analysis uses the built-in return-scaling
                heuristic.
            crisis_hedge_config: Optional hedge configuration. When provided,
                crisis scenarios compute both unhedged and hedged drawdowns
                using VIX-adaptive position scaling from CrisisHedgeController.

        Returns:
            Dict with keys: monte_carlo, crisis_scenarios, sensitivity, summary.
        """
        logger.info(
            "StressTester: running full suite (%d returns, %d MC paths, block=%d)",
            len(self.returns), self.n_simulations, self.block_size,
        )

        mc = self.run_monte_carlo()
        crisis = self.run_crisis_scenarios(crisis_hedge_config=crisis_hedge_config)
        sensitivity = self.run_sensitivity_analysis(param_sweep_fn)

        summary = self._build_summary(mc, crisis, sensitivity)

        return {
            "monte_carlo": mc,
            "crisis_scenarios": crisis,
            "sensitivity": sensitivity,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Monte Carlo (block bootstrap)
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        horizon_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation using block-bootstrap resampling.

        Block bootstrap preserves short-term autocorrelation in daily returns
        (volatility clustering, mean reversion) that i.i.d. resampling destroys.

        Args:
            horizon_days: Simulation horizon. Defaults to len(daily_returns).

        Returns:
            Dict with terminal_wealth distribution, drawdown stats, percentiles.
        """
        n = len(self.returns)
        if n == 0:
            return self._empty_mc_result()

        horizon = horizon_days or n
        block = min(self.block_size, n)

        terminal_values = np.empty(self.n_simulations)
        max_drawdowns = np.empty(self.n_simulations)
        sharpe_ratios = np.empty(self.n_simulations)
        all_paths: List[np.ndarray] = []

        # Store a subset of paths for plotting (max 200 to keep payload small)
        store_paths = min(self.n_simulations, 200)

        for i in range(self.n_simulations):
            sim_returns = self._block_bootstrap(horizon, block)
            equity = _returns_to_equity(sim_returns, self.starting_capital)

            terminal_values[i] = equity[-1]
            max_drawdowns[i] = _max_drawdown(equity)
            sharpe_ratios[i] = _sharpe_ratio(sim_returns)

            if i < store_paths:
                all_paths.append(equity.tolist())

        # Compute percentiles
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        terminal_pcts = {
            f"p{p}": round(_percentile_safe(terminal_values, p), 2) for p in pcts
        }
        dd_pcts = {
            f"p{p}": round(_percentile_safe(max_drawdowns, p) * 100, 2) for p in pcts
        }

        # Probability of ruin (losing > 50% of capital)
        ruin_threshold = self.starting_capital * 0.50
        prob_ruin = float(np.mean(terminal_values < ruin_threshold))

        # Probability of profit
        prob_profit = float(np.mean(terminal_values > self.starting_capital))

        result = {
            "n_simulations": self.n_simulations,
            "horizon_days": horizon,
            "block_size": block,
            "starting_capital": self.starting_capital,
            "terminal_wealth": {
                "mean": round(float(np.mean(terminal_values)), 2),
                "median": round(float(np.median(terminal_values)), 2),
                "std": round(float(np.std(terminal_values)), 2),
                "min": round(float(np.min(terminal_values)), 2),
                "max": round(float(np.max(terminal_values)), 2),
                "percentiles": terminal_pcts,
            },
            "max_drawdown": {
                "mean_pct": round(float(np.mean(max_drawdowns)) * 100, 2),
                "median_pct": round(float(np.median(max_drawdowns)) * 100, 2),
                "worst_pct": round(float(np.min(max_drawdowns)) * 100, 2),
                "percentiles_pct": dd_pcts,
            },
            "sharpe_ratio": {
                "mean": round(float(np.mean(sharpe_ratios)), 3),
                "median": round(float(np.median(sharpe_ratios)), 3),
                "std": round(float(np.std(sharpe_ratios)), 3),
                "percentiles": {
                    f"p{p}": round(_percentile_safe(sharpe_ratios, p), 3) for p in pcts
                },
            },
            "prob_profit": round(prob_profit, 4),
            "prob_ruin_50pct": round(prob_ruin, 4),
            "sample_paths": all_paths,
        }

        logger.info(
            "Monte Carlo complete: median terminal=$%s, P5 DD=%.1f%%, prob_profit=%.1f%%",
            f"{result['terminal_wealth']['median']:,.0f}",
            result["max_drawdown"]["percentiles_pct"]["p5"],
            result["prob_profit"] * 100,
        )
        return result

    def _block_bootstrap(self, horizon: int, block: int) -> np.ndarray:
        """Generate one simulated return path via block bootstrap.

        Randomly samples contiguous blocks of *block* days from the historical
        returns (with replacement) until *horizon* days are filled.  Blocks
        wrap around at array boundaries.
        """
        n = len(self.returns)
        result = np.empty(horizon)
        pos = 0
        while pos < horizon:
            start = self.rng.randint(0, n)
            end = min(pos + block, horizon)
            length = end - pos
            # Extract block with wrap-around
            indices = np.arange(start, start + length) % n
            result[pos:end] = self.returns[indices]
            pos = end
        return result

    def _empty_mc_result(self) -> Dict[str, Any]:
        """Return a valid but empty MC result when no data is available."""
        return {
            "n_simulations": 0,
            "horizon_days": 0,
            "block_size": self.block_size,
            "starting_capital": self.starting_capital,
            "terminal_wealth": {
                "mean": self.starting_capital, "median": self.starting_capital,
                "std": 0, "min": self.starting_capital, "max": self.starting_capital,
                "percentiles": {},
            },
            "max_drawdown": {
                "mean_pct": 0, "median_pct": 0, "worst_pct": 0, "percentiles_pct": {},
            },
            "sharpe_ratio": {
                "mean": 0, "median": 0, "std": 0, "percentiles": {},
            },
            "prob_profit": 0,
            "prob_ruin_50pct": 0,
            "sample_paths": [],
        }

    # ------------------------------------------------------------------
    # Crisis scenario analysis
    # ------------------------------------------------------------------

    def run_crisis_scenarios(
        self,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        crisis_hedge_config: Optional[CrisisHedgeConfig] = None,
    ) -> List[Dict[str, Any]]:
        """Overlay historical crisis return paths onto the portfolio.

        For each scenario, applies the crisis daily shocks to the current
        portfolio and measures peak-to-trough drawdown, days to recovery
        (using historical mean return), and margin impact.

        When *crisis_hedge_config* is provided, a parallel hedged path is
        computed by interpolating VIX linearly from ``vix_start`` to
        ``vix_peak`` over the crisis duration and scaling each daily shock
        by :meth:`CrisisHedgeController.position_scale_factor`.

        Args:
            scenarios: List of scenario dicts. Defaults to CRISIS_SCENARIOS.
            crisis_hedge_config: Optional hedge configuration for computing
                hedged drawdowns alongside unhedged results.

        Returns:
            List of scenario result dicts.
        """
        scenarios = scenarios or CRISIS_SCENARIOS
        results = []

        hedge_controller: Optional[CrisisHedgeController] = None
        if crisis_hedge_config is not None:
            hedge_controller = CrisisHedgeController(crisis_hedge_config)

        # Historical daily mean return (for recovery estimation)
        if len(self.returns) > 0:
            hist_daily_mean = float(np.mean(self.returns))
        else:
            hist_daily_mean = 0.0

        for scenario in scenarios:
            shocks = np.asarray(scenario["daily_shocks"], dtype=np.float64)
            if len(shocks) == 0:
                continue

            n_days = len(shocks)

            # Apply shocks to portfolio (unhedged)
            equity = _returns_to_equity(shocks, self.starting_capital)
            trough = float(np.min(equity))
            trough_pct = (trough - self.starting_capital) / self.starting_capital
            peak_dd = _max_drawdown(equity)

            # Credit spread specific: estimate additional losses from spread widening.
            # When VIX spikes, short put/call spreads move against us.
            # Heuristic: portfolio beta to crisis = 1.5x for credit spreads
            # (short gamma position suffers more than underlying)
            spread_beta = 1.5
            adjusted_trough_pct = trough_pct * spread_beta
            adjusted_trough = self.starting_capital * (1 + adjusted_trough_pct)

            # Estimate days to recover (from trough back to starting capital)
            if hist_daily_mean > 0 and adjusted_trough_pct < 0:
                recovery_return_needed = (self.starting_capital / adjusted_trough) - 1
                days_to_recover = int(
                    math.ceil(math.log(1 + recovery_return_needed) / math.log(1 + hist_daily_mean))
                )
            else:
                days_to_recover = None  # cannot estimate

            # VIX impact on credit spread P&L
            vix_start = scenario.get("vix_start", 15)
            vix_peak = scenario.get("vix_peak", 40)
            vix_multiplier = vix_peak / max(vix_start, 1)

            # ── Hedged path (VIX-adaptive scaling) ────────────────────────
            hedged_portfolio_drawdown_pct = None
            hedged_trough_value = None
            hedged_equity_path = None
            hedged_peak_dd = None

            if hedge_controller is not None:
                hedged_shocks = np.empty_like(shocks)
                for day_idx in range(n_days):
                    # Linearly interpolate VIX from start to peak over the crisis
                    t = day_idx / max(n_days - 1, 1)
                    vix_today = vix_start + (vix_peak - vix_start) * t
                    scale = hedge_controller.position_scale_factor(vix=vix_today)
                    hedged_shocks[day_idx] = shocks[day_idx] * scale

                hedged_equity = _returns_to_equity(hedged_shocks, self.starting_capital)
                hedged_trough = float(np.min(hedged_equity))
                hedged_trough_pct = (hedged_trough - self.starting_capital) / self.starting_capital
                hedged_peak_dd = _max_drawdown(hedged_equity)

                # Apply same spread_beta to hedged path
                hedged_adjusted_trough_pct = hedged_trough_pct * spread_beta
                hedged_portfolio_drawdown_pct = round(hedged_adjusted_trough_pct * 100, 2)
                hedged_trough_value = round(
                    self.starting_capital * (1 + hedged_adjusted_trough_pct), 2,
                )
                hedged_equity_path = hedged_equity.tolist()

            result_entry = {
                "name": scenario["name"],
                "description": scenario["description"],
                "n_days": n_days,
                "underlying_drawdown_pct": round(trough_pct * 100, 2),
                "portfolio_drawdown_pct": round(adjusted_trough_pct * 100, 2),
                "max_drawdown_pct": round(peak_dd * 100, 2),
                "trough_value": round(adjusted_trough, 2),
                "spread_beta": spread_beta,
                "vix_start": vix_start,
                "vix_peak": vix_peak,
                "vix_multiplier": round(vix_multiplier, 2),
                "estimated_recovery_days": days_to_recover,
                "equity_path": equity.tolist(),
                # Hedged results (None when no hedge config provided)
                "hedged_portfolio_drawdown_pct": hedged_portfolio_drawdown_pct,
                "hedged_trough_value": hedged_trough_value,
                "hedged_equity_path": hedged_equity_path,
            }

            results.append(result_entry)

            if hedge_controller is not None:
                logger.info(
                    "Crisis '%s': unhedged DD=%.1f%%, hedged DD=%.1f%%, recovery=%s days",
                    scenario["name"],
                    adjusted_trough_pct * 100,
                    hedged_portfolio_drawdown_pct,
                    days_to_recover or "N/A",
                )
            else:
                logger.info(
                    "Crisis '%s': underlying DD=%.1f%%, portfolio DD=%.1f%%, recovery=%s days",
                    scenario["name"],
                    trough_pct * 100,
                    adjusted_trough_pct * 100,
                    days_to_recover or "N/A",
                )

        return results

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    # Default parameter sweep ranges (matching config_exp154.yaml structure)
    DEFAULT_SWEEPS: Dict[str, Dict[str, Any]] = {
        "position_size_pct": {
            "label": "Position Size (% of account)",
            "values": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
            "baseline": 5.0,
            "description": "Risk per trade as pct of account (risk.max_risk_per_trade)",
        },
        "stop_loss_multiplier": {
            "label": "Stop Loss Multiplier",
            "values": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            "baseline": 3.5,
            "description": "Stop loss as multiple of credit received",
        },
        "iv_rank_threshold": {
            "label": "IV Rank Entry Threshold",
            "values": [0, 5, 10, 15, 20, 30, 40, 50],
            "baseline": 12,
            "description": "Minimum IV rank to enter a trade (strategy.min_iv_rank)",
        },
        "profit_target_pct": {
            "label": "Profit Target (%)",
            "values": [25, 40, 50, 60, 75, 90],
            "baseline": 50,
            "description": "Close at this % of max profit (risk.profit_target)",
        },
        "spread_width": {
            "label": "Spread Width ($)",
            "values": [2.5, 5.0, 7.5, 10.0, 15.0, 20.0],
            "baseline": 5.0,
            "description": "Width between short and long strikes",
        },
    }

    def run_sensitivity_analysis(
        self,
        param_sweep_fn: Optional[Callable] = None,
        sweeps: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Sweep key strategy parameters and measure impact on risk/return.

        When *param_sweep_fn* is provided, it is called for each (param, value)
        pair and should return a daily returns array from a full backtest re-run.

        When *param_sweep_fn* is None (default), a heuristic scaling model is
        used that adjusts the historical returns to approximate the effect:
          - position_size: scales returns linearly
          - stop_loss: truncates losses beyond the multiplier
          - iv_rank_threshold: filters out a fraction of lower-quality trades
          - profit_target: caps wins at target fraction
          - spread_width: scales both risk and reward

        Args:
            param_sweep_fn: Optional backtest re-run function.
            sweeps: Override parameter definitions. Defaults to DEFAULT_SWEEPS.

        Returns:
            Dict keyed by param name, each containing a list of
            {value, sharpe, max_dd_pct, cagr_pct, calmar} dicts.
        """
        sweeps = sweeps or self.DEFAULT_SWEEPS
        results: Dict[str, Any] = {}

        for param_name, sweep_cfg in sweeps.items():
            param_results = []
            baseline = sweep_cfg["baseline"]

            for value in sweep_cfg["values"]:
                if param_sweep_fn is not None:
                    sim_returns = np.asarray(
                        param_sweep_fn(param_name, value), dtype=np.float64,
                    )
                else:
                    sim_returns = self._heuristic_param_adjustment(
                        param_name, value, baseline,
                    )

                equity = _returns_to_equity(sim_returns, self.starting_capital)
                sharpe = _sharpe_ratio(sim_returns)
                dd = _max_drawdown(equity)
                cagr = _cagr(equity)
                calmar = _calmar_ratio(equity)

                param_results.append({
                    "value": value,
                    "sharpe": round(sharpe, 3),
                    "max_dd_pct": round(dd * 100, 2),
                    "cagr_pct": round(cagr * 100, 2),
                    "calmar": round(calmar, 3),
                    "terminal_value": round(float(equity[-1]), 2),
                    "is_baseline": abs(value - baseline) < 1e-9,
                })

            results[param_name] = {
                "label": sweep_cfg["label"],
                "description": sweep_cfg["description"],
                "baseline": baseline,
                "results": param_results,
            }

            logger.info(
                "Sensitivity '%s': swept %d values, baseline Sharpe=%.3f",
                param_name,
                len(param_results),
                next(
                    (r["sharpe"] for r in param_results if r["is_baseline"]),
                    0,
                ),
            )

        return results

    def _heuristic_param_adjustment(
        self,
        param_name: str,
        value: float,
        baseline: float,
    ) -> np.ndarray:
        """Approximate the effect of a parameter change on daily returns.

        This is a heuristic model — not a substitute for a full backtest re-run.
        It provides directionally correct sensitivity curves for rapid analysis.
        """
        returns = self.returns.copy()

        if baseline == 0:
            return returns

        ratio = value / baseline

        if param_name == "position_size_pct":
            # Linear scaling: doubling position size doubles returns and risk
            returns = returns * ratio

        elif param_name == "stop_loss_multiplier":
            # Tighter stops truncate large losses but also clip recoveries.
            # Model: cap the worst daily losses proportionally.
            if ratio < 1.0:
                # Tighter stop → losses are clipped closer to zero
                loss_cap = np.percentile(returns[returns < 0], 100 * (1 - ratio)) if np.any(returns < 0) else 0
                returns = np.maximum(returns, loss_cap)
            elif ratio > 1.0:
                # Wider stop → allow bigger drawdowns (amplify worst losses slightly)
                mask = returns < 0
                returns[mask] = returns[mask] * (1 + (ratio - 1) * 0.3)

        elif param_name == "iv_rank_threshold":
            # Higher threshold filters out more trades.  Model: randomly remove
            # a fraction of trades proportional to how much we raised the bar.
            if value > baseline and len(returns) > 0:
                # Fraction of trades that would be filtered out
                filter_frac = min(0.8, (value - baseline) / 100.0)
                mask = self.rng.random(len(returns)) > filter_frac
                # Filtered-out days become zero-return (no trade taken)
                returns = np.where(mask, returns, 0.0)
            elif value < baseline and len(returns) > 0:
                # Lower threshold admits more (potentially lower-quality) trades.
                # Model: add noise to represent less selective entries.
                noise_scale = abs(baseline - value) / 100.0 * np.std(returns)
                returns = returns + self.rng.normal(0, noise_scale, len(returns))

        elif param_name == "profit_target_pct":
            # Lower target caps gains earlier; higher target lets winners run.
            if len(returns) > 0:
                cap_ratio = value / baseline
                positive_mask = returns > 0
                returns[positive_mask] = returns[positive_mask] * min(cap_ratio, 1.5)

        elif param_name == "spread_width":
            # Wider spreads scale both max risk and credit.
            # Net effect is roughly linear on returns with diminishing benefit.
            returns = returns * (ratio ** 0.7)

        else:
            logger.debug("No heuristic for param '%s'; returning raw returns", param_name)

        return returns

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        mc: Dict[str, Any],
        crisis: List[Dict[str, Any]],
        sensitivity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a top-level summary suitable for dashboard display."""
        # Historical baseline metrics
        if len(self.returns) > 0:
            equity = _returns_to_equity(self.returns, self.starting_capital)
            hist_sharpe = _sharpe_ratio(self.returns)
            hist_dd = _max_drawdown(equity)
            hist_cagr = _cagr(equity)
            hist_calmar = _calmar_ratio(equity)
        else:
            hist_sharpe = hist_dd = hist_cagr = hist_calmar = 0.0

        # Worst crisis impact (use hedged DD when available for effective worst)
        worst_crisis = min(crisis, key=lambda c: c["portfolio_drawdown_pct"]) if crisis else None
        worst_crisis_hedged = None
        if crisis and any(c.get("hedged_portfolio_drawdown_pct") is not None for c in crisis):
            hedged_scenarios = [c for c in crisis if c.get("hedged_portfolio_drawdown_pct") is not None]
            if hedged_scenarios:
                worst_crisis_hedged = min(hedged_scenarios, key=lambda c: c["hedged_portfolio_drawdown_pct"])

        # Find the sensitivity param with largest Sharpe range
        max_sharpe_range = 0.0
        most_sensitive_param = None
        for param_name, param_data in sensitivity.items():
            sharpes = [r["sharpe"] for r in param_data["results"]]
            if sharpes:
                s_range = max(sharpes) - min(sharpes)
                if s_range > max_sharpe_range:
                    max_sharpe_range = s_range
                    most_sensitive_param = param_name

        return {
            "historical": {
                "n_days": len(self.returns),
                "starting_capital": self.starting_capital,
                "sharpe": round(hist_sharpe, 3),
                "max_drawdown_pct": round(hist_dd * 100, 2),
                "cagr_pct": round(hist_cagr * 100, 2),
                "calmar": round(hist_calmar, 3),
            },
            "monte_carlo_confidence": {
                "p5_terminal": mc["terminal_wealth"]["percentiles"].get("p5", 0),
                "p50_terminal": mc["terminal_wealth"]["percentiles"].get("p50", 0),
                "p95_terminal": mc["terminal_wealth"]["percentiles"].get("p95", 0),
                "prob_profit_pct": round(mc["prob_profit"] * 100, 2),
                "prob_ruin_pct": round(mc["prob_ruin_50pct"] * 100, 2),
                "median_max_dd_pct": mc["max_drawdown"]["median_pct"],
            },
            "worst_crisis": {
                "name": worst_crisis["name"] if worst_crisis else "N/A",
                "portfolio_drawdown_pct": worst_crisis["portfolio_drawdown_pct"] if worst_crisis else 0,
                "hedged_portfolio_drawdown_pct": (
                    worst_crisis_hedged["hedged_portfolio_drawdown_pct"]
                    if worst_crisis_hedged else None
                ),
                "estimated_recovery_days": worst_crisis["estimated_recovery_days"] if worst_crisis else None,
            },
            "most_sensitive_parameter": most_sensitive_param,
            "risk_rating": self._compute_risk_rating(mc, crisis),
        }

    def _compute_risk_rating(
        self,
        mc: Dict[str, Any],
        crisis: List[Dict[str, Any]],
    ) -> str:
        """Assign a qualitative risk rating based on stress test results.

        Returns one of: "LOW", "MODERATE", "HIGH", "CRITICAL".
        """
        score = 0

        # MC probability of ruin
        prob_ruin = mc.get("prob_ruin_50pct", 0)
        if prob_ruin > 0.10:
            score += 3
        elif prob_ruin > 0.05:
            score += 2
        elif prob_ruin > 0.01:
            score += 1

        # MC median drawdown
        median_dd = abs(mc.get("max_drawdown", {}).get("median_pct", 0))
        if median_dd > 30:
            score += 3
        elif median_dd > 20:
            score += 2
        elif median_dd > 10:
            score += 1

        # Worst crisis drawdown (use hedged DD when available)
        if crisis:
            dds = []
            for c in crisis:
                h = c.get("hedged_portfolio_drawdown_pct")
                dds.append(h if h is not None else c["portfolio_drawdown_pct"])
            worst_dd = abs(min(dds))
            if worst_dd > 60:
                score += 3
            elif worst_dd > 40:
                score += 2
            elif worst_dd > 20:
                score += 1

        # MC probability of profit
        prob_profit = mc.get("prob_profit", 1)
        if prob_profit < 0.50:
            score += 2
        elif prob_profit < 0.70:
            score += 1

        if score >= 7:
            return "CRITICAL"
        elif score >= 4:
            return "HIGH"
        elif score >= 2:
            return "MODERATE"
        return "LOW"
