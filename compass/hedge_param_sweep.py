"""
compass/hedge_param_sweep.py — Grid search over CrisisHedgeConfig parameters.

Sweeps vix_scale_floor, vix_scale_ceiling, base_stop_multiplier, and
high_vol_regime_scale for both EXP-400 and EXP-401 trade datasets.
For each combo, applies CrisisHedgeController to the trade-level PnL,
runs a Monte Carlo block-bootstrap stress test, and records:

  - MC P5 max drawdown (%)
  - Hedged Sharpe ratio (median across MC paths)
  - Annual return (%) derived from terminal wealth

Outputs:
  - experiments/sweep_results.csv          — full grid, sorted by MC P5 DD
  - experiments/sweep_summary.md           — top configs + analysis

Usage::

    python3 -m compass.hedge_param_sweep                   # full sweep
    python3 -m compass.hedge_param_sweep --n-sims 500      # fast preview
    python3 -m compass.hedge_param_sweep --experiment 400   # EXP-400 only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from compass.crisis_hedge import CrisisHedgeConfig, CrisisHedgeController
from compass.stress_test import StressTester

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STARTING_CAPITAL = 100_000

# ── Default grid values ────────────────────────────────────────────────────

DEFAULT_VIX_FLOORS = [12.0, 14.0, 16.0, 18.0, 20.0]
DEFAULT_VIX_CEILINGS = [35.0, 38.0, 42.0, 46.0, 50.0]
DEFAULT_BASE_STOPS = [1.5, 2.0, 2.5, 3.0, 3.5]
DEFAULT_HV_SCALES = [0.05, 0.10, 0.15, 0.25]


# ── Data loading (reuses run_stress_test logic) ───────────────────────────


def load_daily_returns(csv_path: Path, starting_capital: float) -> pd.Series:
    """Load trade CSV and convert to daily return series."""
    df = pd.read_csv(csv_path, parse_dates=["exit_date", "entry_date"])
    daily_pnl = df.groupby("exit_date")["pnl"].sum()
    start = df["entry_date"].min()
    end = df["exit_date"].max()
    all_days = pd.bdate_range(start=start, end=end)
    daily_pnl = daily_pnl.reindex(all_days, fill_value=0.0)
    return daily_pnl / starting_capital


def load_daily_returns_hedged(
    csv_path: Path,
    starting_capital: float,
    controller: CrisisHedgeController,
) -> pd.Series:
    """Load trade CSV and apply hedge controller to each trade's PnL."""
    base_stop = controller.cfg.base_stop_multiplier
    df = pd.read_csv(csv_path, parse_dates=["exit_date", "entry_date"])

    hedged_pnl = []
    for _, row in df.iterrows():
        vix = float(row.get("vix", 20.0)) if not pd.isna(row.get("vix", float("nan"))) else 20.0
        regime = str(row.get("regime", "neutral")) if not pd.isna(row.get("regime", float("nan"))) else "neutral"
        pnl = float(row["pnl"])

        scale = controller.position_scale_factor(vix=vix, regime=regime)

        if pnl < 0 and str(row.get("exit_reason", "")).lower() == "stop_loss":
            stop_mult = controller.stop_loss_multiplier(vix=vix, regime=regime)
            tighten_ratio = stop_mult / base_stop
            pnl = max(pnl * tighten_ratio, pnl)

        hedged_pnl.append(pnl * scale)

    df["hedged_pnl"] = hedged_pnl
    daily_pnl = df.groupby("exit_date")["hedged_pnl"].sum()
    start = df["entry_date"].min()
    end = df["exit_date"].max()
    all_days = pd.bdate_range(start=start, end=end)
    daily_pnl = daily_pnl.reindex(all_days, fill_value=0.0)
    return daily_pnl / starting_capital


# ── Single-config evaluation ──────────────────────────────────────────────


@dataclass
class SweepResult:
    """Metrics from one parameter-combo evaluation."""

    experiment: str
    vix_floor: float
    vix_ceiling: float
    base_stop: float
    hv_scale: float
    mc_p5_dd: float       # MC 5th-percentile max drawdown (%, negative)
    hedged_sharpe: float   # Median Sharpe across MC paths
    annual_return_pct: float
    passes_30pct: bool     # abs(mc_p5_dd) <= 30


def evaluate_config(
    daily_returns_hedged: np.ndarray,
    experiment: str,
    config: CrisisHedgeConfig,
    n_simulations: int = 1_000,
    starting_capital: float = STARTING_CAPITAL,
) -> SweepResult:
    """Run MC stress test on pre-hedged returns and extract metrics."""
    tester = StressTester(
        daily_returns_hedged,
        starting_capital=starting_capital,
        n_simulations=n_simulations,
        block_size=5,
        seed=42,
    )
    mc = tester.run_monte_carlo()

    mc_p5_dd = mc["max_drawdown"]["percentiles_pct"].get("p5", 0.0)
    hedged_sharpe = mc["sharpe_ratio"]["median"]

    # Annual return from median terminal wealth
    median_terminal = mc["terminal_wealth"]["median"]
    horizon_days = mc["horizon_days"]
    years = horizon_days / 252.0 if horizon_days > 0 else 1.0
    if median_terminal > 0 and starting_capital > 0:
        total_return = median_terminal / starting_capital
        annual_return_pct = ((total_return ** (1.0 / years)) - 1) * 100
    else:
        annual_return_pct = 0.0

    return SweepResult(
        experiment=experiment,
        vix_floor=config.vix_scale_floor,
        vix_ceiling=config.vix_scale_ceiling,
        base_stop=config.base_stop_multiplier,
        hv_scale=config.high_vol_regime_scale,
        mc_p5_dd=mc_p5_dd,
        hedged_sharpe=round(hedged_sharpe, 3),
        annual_return_pct=round(annual_return_pct, 2),
        passes_30pct=abs(mc_p5_dd) <= 30.0,
    )


# ── Grid sweep ────────────────────────────────────────────────────────────


def build_grid(
    vix_floors: Sequence[float] = DEFAULT_VIX_FLOORS,
    vix_ceilings: Sequence[float] = DEFAULT_VIX_CEILINGS,
    base_stops: Sequence[float] = DEFAULT_BASE_STOPS,
    hv_scales: Sequence[float] = DEFAULT_HV_SCALES,
) -> List[CrisisHedgeConfig]:
    """Generate all valid parameter combinations.

    Filters out invalid combos where vix_floor >= vix_ceiling.
    Sets vix_stop thresholds proportionally to the scale thresholds:
      vix_stop_floor = vix_scale_floor
      vix_stop_ceiling = vix_scale_floor + 0.6 * (vix_scale_ceiling - vix_scale_floor)
      min_stop_multiplier = max(1.0, base_stop - 1.5)
    """
    configs: List[CrisisHedgeConfig] = []
    for floor, ceiling, stop, hv in itertools.product(
        vix_floors, vix_ceilings, base_stops, hv_scales
    ):
        if floor >= ceiling:
            continue
        # Derive stop thresholds from scale thresholds
        span = ceiling - floor
        stop_ceiling = floor + 0.6 * span
        min_stop = max(1.0, stop - 1.5)
        configs.append(CrisisHedgeConfig(
            vix_scale_floor=floor,
            vix_scale_ceiling=ceiling,
            vix_stop_floor=floor,
            vix_stop_ceiling=round(stop_ceiling, 1),
            base_stop_multiplier=stop,
            min_stop_multiplier=round(min_stop, 1),
            high_vol_regime_scale=hv,
            log_decisions=False,
        ))
    return configs


def run_sweep(
    csv_path: Path,
    experiment: str,
    configs: List[CrisisHedgeConfig],
    n_simulations: int = 1_000,
) -> List[SweepResult]:
    """Run the full parameter sweep for one experiment.

    Args:
        csv_path: Path to the trade-level CSV (training_data_exp*.csv).
        experiment: Label (e.g. "EXP-400").
        configs: List of CrisisHedgeConfig combos to evaluate.
        n_simulations: MC paths per combo.

    Returns:
        List of SweepResult sorted by mc_p5_dd (best first, i.e. least negative).
    """
    results: List[SweepResult] = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        controller = CrisisHedgeController(cfg)
        hedged_returns = load_daily_returns_hedged(csv_path, STARTING_CAPITAL, controller)

        result = evaluate_config(
            hedged_returns.values,
            experiment=experiment,
            config=cfg,
            n_simulations=n_simulations,
        )
        results.append(result)

        if i % 25 == 0 or i == total:
            logger.info(
                "%s: %d/%d combos evaluated (latest P5 DD=%.1f%%)",
                experiment, i, total, result.mc_p5_dd,
            )

    # Sort by P5 DD descending (least negative = best)
    results.sort(key=lambda r: r.mc_p5_dd, reverse=True)
    return results


# ── Output ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "experiment", "vix_floor", "vix_ceiling", "base_stop", "hv_scale",
    "mc_p5_dd", "hedged_sharpe", "annual_return_pct", "passes_30pct",
]


def write_csv(results: List[SweepResult], path: Path) -> None:
    """Write sweep results to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "experiment": r.experiment,
                "vix_floor": r.vix_floor,
                "vix_ceiling": r.vix_ceiling,
                "base_stop": r.base_stop,
                "hv_scale": r.hv_scale,
                "mc_p5_dd": round(r.mc_p5_dd, 2),
                "hedged_sharpe": r.hedged_sharpe,
                "annual_return_pct": r.annual_return_pct,
                "passes_30pct": r.passes_30pct,
            })
    logger.info("Wrote %d rows to %s", len(results), path)


def write_summary_md(results: List[SweepResult], path: Path) -> None:
    """Write a markdown summary of the sweep results."""
    path.parent.mkdir(parents=True, exist_ok=True)

    experiments = sorted(set(r.experiment for r in results))
    lines = [
        "# Hedge Parameter Sweep Results",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total combos evaluated: {len(results)}",
        "",
    ]

    for exp in experiments:
        exp_results = [r for r in results if r.experiment == exp]
        passing = [r for r in exp_results if r.passes_30pct]
        top10 = exp_results[:10]
        best = exp_results[0] if exp_results else None

        lines.append(f"## {exp}")
        lines.append("")
        lines.append(f"- Combos tested: {len(exp_results)}")
        lines.append(f"- Passing (MC P5 DD <= 30%): {len(passing)}/{len(exp_results)}")
        if best:
            lines.append(f"- Best MC P5 DD: {best.mc_p5_dd:.1f}%")
            lines.append(
                f"- Best config: floor={best.vix_floor}, ceiling={best.vix_ceiling}, "
                f"stop={best.base_stop}, hv_scale={best.hv_scale}"
            )
        lines.append("")

        # Top 10 table
        lines.append("### Top 10 Configs")
        lines.append("")
        lines.append(
            "| VIX Floor | VIX Ceiling | Base Stop | HV Scale | MC P5 DD | Sharpe | Ann. Return | Pass |"
        )
        lines.append(
            "|-----------|-------------|-----------|----------|----------|--------|-------------|------|"
        )
        for r in top10:
            tick = "PASS" if r.passes_30pct else "FAIL"
            lines.append(
                f"| {r.vix_floor:.0f} | {r.vix_ceiling:.0f} | {r.base_stop:.1f} | "
                f"{r.hv_scale:.2f} | {r.mc_p5_dd:.1f}% | {r.hedged_sharpe:.3f} | "
                f"{r.annual_return_pct:.1f}% | {tick} |"
            )
        lines.append("")

        # Parameter sensitivity analysis
        if exp_results:
            lines.append("### Parameter Sensitivity")
            lines.append("")
            for param_name, getter in [
                ("vix_floor", lambda r: r.vix_floor),
                ("vix_ceiling", lambda r: r.vix_ceiling),
                ("base_stop", lambda r: r.base_stop),
                ("hv_scale", lambda r: r.hv_scale),
            ]:
                vals = sorted(set(getter(r) for r in exp_results))
                if len(vals) < 2:
                    continue
                lines.append(f"**{param_name}:**")
                for v in vals:
                    subset = [r for r in exp_results if getter(r) == v]
                    if not subset:
                        continue
                    avg_dd = sum(r.mc_p5_dd for r in subset) / len(subset)
                    avg_sharpe = sum(r.hedged_sharpe for r in subset) / len(subset)
                    pct_pass = sum(1 for r in subset if r.passes_30pct) / len(subset) * 100
                    lines.append(
                        f"  - {v}: avg P5 DD={avg_dd:.1f}%, avg Sharpe={avg_sharpe:.3f}, "
                        f"pass rate={pct_pass:.0f}%"
                    )
                lines.append("")

    path.write_text("\n".join(lines))
    logger.info("Wrote summary to %s", path)


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Hedge parameter sweep")
    parser.add_argument("--n-sims", type=int, default=1_000,
                        help="MC simulations per combo (default 1000)")
    parser.add_argument("--experiment", type=str, default="both",
                        choices=["400", "401", "both"],
                        help="Which experiment to sweep")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "experiments"),
                        help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    configs = build_grid()
    logger.info("Grid: %d parameter combos", len(configs))

    all_results: List[SweepResult] = []

    if args.experiment in ("400", "both"):
        csv_400 = ROOT / "compass" / "training_data_exp400.csv"
        logger.info("Sweeping EXP-400 (%s)", csv_400)
        results_400 = run_sweep(csv_400, "EXP-400", configs, args.n_sims)
        all_results.extend(results_400)

    if args.experiment in ("401", "both"):
        csv_401 = ROOT / "compass" / "training_data_exp401.csv"
        logger.info("Sweeping EXP-401 (%s)", csv_401)
        results_401 = run_sweep(csv_401, "EXP-401", configs, args.n_sims)
        all_results.extend(results_401)

    out_dir = Path(args.output_dir)
    write_csv(all_results, out_dir / "sweep_results.csv")
    write_summary_md(all_results, out_dir / "sweep_summary.md")

    # Print quick summary
    for exp in sorted(set(r.experiment for r in all_results)):
        exp_r = [r for r in all_results if r.experiment == exp]
        best = exp_r[0] if exp_r else None
        n_pass = sum(1 for r in exp_r if r.passes_30pct)
        if best:
            print(
                f"{exp}: best P5 DD={best.mc_p5_dd:.1f}% "
                f"(floor={best.vix_floor}, ceil={best.vix_ceiling}, "
                f"stop={best.base_stop}, hv={best.hv_scale}) "
                f"— {n_pass}/{len(exp_r)} pass"
            )


if __name__ == "__main__":
    main()
