#!/usr/bin/env python3
"""
Mass Trade Harvester V2 — Diverse ML Training Data Collection

Runs ~63 different parameter configs across 5 strategies (CS, IC, SS, DS, MS)
through the portfolio backtester for 6 years (2020-2025). Each config varies
ENTRY parameters (target_dte, otm_pct, direction, etc.) to produce different
trade entry dates/strikes, maximizing trade diversity.

Output:
  compass/training_data_v2.csv          — 3-5K enriched trades (40+ columns)
  compass/harvest_quality_report.md     — Quality report

Usage:
    cd /home/node/openclaw/workspace/pilotai-credit-spreads
    PYTHONPATH=. python3 compass/harvest_trades_v2.py --dry-run      # verify configs
    PYTHONPATH=. python3 compass/harvest_trades_v2.py                # full run
    PYTHONPATH=. python3 compass/harvest_trades_v2.py --strategies cs ic  # subset
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from compass.collect_training_data import (
    _load_full_market_data,
    enrich_trades,
    load_champion_params,
)
from engine.portfolio_backtester import PortfolioBacktester
from strategies import STRATEGY_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("harvest_v2")

# ── Constants ──────────────────────────────────────────────────────────────
COMPASS_DIR = ROOT / "compass"
CHAMPION_PATH = ROOT / "configs" / "champion.json"
OUTPUT_CSV = COMPASS_DIR / "training_data_v2.csv"
OUTPUT_REPORT = COMPASS_DIR / "harvest_quality_report.md"
TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
YEARS = list(range(2020, 2026))

# Short codes for strategy names
STRATEGY_SHORT = {
    "credit_spread": "CS",
    "iron_condor": "IC",
    "straddle_strangle": "SS",
    "debit_spread": "DS",
    "momentum_swing": "MS",
}


# ═══════════════════════════════════════════════════════════════════════════
# Parameter Grid Generators
# ═══════════════════════════════════════════════════════════════════════════

def _base_params(strategy_name: str) -> Dict:
    """Get champion params as base, falling back to strategy defaults."""
    champ = load_champion_params()
    if strategy_name in champ:
        return dict(champ[strategy_name])
    cls = STRATEGY_REGISTRY[strategy_name]
    return cls.get_default_params()


def _cs_grid() -> List[Tuple[str, Dict]]:
    """Credit Spread: 32 configs.

    Cross: target_dte=[15,21,30,45] × otm_pct=[0.02,0.03,0.05,0.07] × direction=[both,regime_adaptive]
    Fixed: spread_width=12, trend_ma_period=80, scan_weekday=any
    """
    base = _base_params("credit_spread")
    configs = []
    for dte, otm, direction in product(
        [15, 21, 30, 45],
        [0.02, 0.03, 0.05, 0.07],
        ["both", "regime_adaptive"],
    ):
        p = dict(base)
        p["target_dte"] = dte
        p["min_dte"] = min(dte, p.get("min_dte", dte))
        p["otm_pct"] = otm
        p["direction"] = direction
        label = f"CS_dte{dte}_otm{int(otm*100)}_dir{direction[:3]}"
        configs.append((label, p))
    return configs


def _ic_grid() -> List[Tuple[str, Dict]]:
    """Iron Condor: 12 configs.

    Cross: rsi_range=[(25,75),(30,70),(35,65),(40,60)] × min_iv_rank=[0,20,45]
    Fixed: target_dte=30, OTM=champion defaults
    """
    base = _base_params("iron_condor")
    configs = []
    for (rsi_lo, rsi_hi), iv_rank in product(
        [(25, 75), (30, 70), (35, 65), (40, 60)],
        [0, 20, 45],
    ):
        p = dict(base)
        p["rsi_min"] = rsi_lo
        p["rsi_max"] = rsi_hi
        p["min_iv_rank"] = float(iv_rank)
        label = f"IC_rsi{rsi_lo}_{rsi_hi}_iv{iv_rank}"
        configs.append((label, p))
    return configs


def _ss_grid() -> List[Tuple[str, Dict]]:
    """Straddle/Strangle: 9 configs.

    Cross: mode=[long_pre_event,short_post_event,both] × event_types=[all,fomc_only,fomc_cpi]
    Fixed: target_dte from SS defaults
    """
    base = _base_params("straddle_strangle")
    configs = []
    for mode, events in product(
        ["long_pre_event", "short_post_event", "both"],
        ["all", "fomc_only", "fomc_cpi"],
    ):
        p = dict(base)
        p["mode"] = mode
        p["event_types"] = events
        label = f"SS_{mode[:5]}_{events[:4]}"
        configs.append((label, p))
    return configs


def _ds_grid() -> List[Tuple[str, Dict]]:
    """Debit Spread: 6 configs.

    Cross: trend_ma_period=[20,34,50] × target_dte=[7,12]
    Fixed: direction=trend_following, other params from champion
    """
    base = _base_params("debit_spread")
    configs = []
    for ma, dte in product([20, 34, 50], [7, 12]):
        p = dict(base)
        p["trend_ma_period"] = ma
        p["target_dte"] = dte
        label = f"DS_ma{ma}_dte{dte}"
        configs.append((label, p))
    return configs


def _ms_grid() -> List[Tuple[str, Dict]]:
    """Momentum Swing: 4 configs.

    Cross: ema_fast/slow=[(8,21),(9,34)] × min_adx=[20,28]
    Fixed: mode=itm_debit_spread from champion
    """
    base = _base_params("momentum_swing")
    configs = []
    for (fast, slow), adx in product([(8, 21), (9, 34)], [20, 28]):
        p = dict(base)
        p["ema_fast"] = fast
        p["ema_slow"] = slow
        p["min_adx"] = float(adx)
        label = f"MS_ema{fast}_{slow}_adx{adx}"
        configs.append((label, p))
    return configs


# Map of strategy name -> grid generator
GRID_GENERATORS = {
    "cs": ("credit_spread", _cs_grid),
    "ic": ("iron_condor", _ic_grid),
    "ss": ("straddle_strangle", _ss_grid),
    "ds": ("debit_spread", _ds_grid),
    "ms": ("momentum_swing", _ms_grid),
}


# ═══════════════════════════════════════════════════════════════════════════
# Backtest Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_single_config(
    strategy_name: str,
    params: Dict,
    year: int,
) -> Tuple[PortfolioBacktester, Dict]:
    """Run a single strategy config for one year."""
    cls = STRATEGY_REGISTRY[strategy_name]
    strategy = cls(dict(params))

    bt = PortfolioBacktester(
        strategies=[(strategy_name, strategy)],
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=10,
        max_positions_per_strategy=5,
    )
    results = bt.run()
    combined = results.get("combined", results)
    return bt, combined


# ═══════════════════════════════════════════════════════════════════════════
# Extended Enrichment
# ═══════════════════════════════════════════════════════════════════════════

def _classify_strategy_type_v2(strat_name: str) -> str:
    """Map strategy name to short code — handles all 5 strategies."""
    lower = strat_name.lower()
    if "iron" in lower or "condor" in lower:
        return "IC"
    if "straddle" in lower or "strangle" in lower:
        return "SS"
    if "debit" in lower:
        return "DS"
    if "momentum" in lower or "swing" in lower:
        return "MS"
    return "CS"


def extend_enrichment(
    trades: List[Dict],
    config_label: str,
    vix_series: pd.Series,
) -> List[Dict]:
    """Add extra columns beyond what enrich_trades() provides."""
    for row in trades:
        # Config label
        row["config_label"] = config_label

        # Underlying (always SPY for now)
        row["underlying"] = "SPY"

        # Fix strategy_type for DS/MS (base enrich_trades only maps CS/IC/SS)
        if config_label.startswith("DS_"):
            row["strategy_type"] = "DS"
        elif config_label.startswith("MS_"):
            row["strategy_type"] = "MS"

        # Credit-to-width ratio
        nc = row.get("net_credit")
        sw = row.get("spread_width")
        if nc is not None and sw is not None and sw > 0:
            row["credit_to_width_ratio"] = round(nc / sw, 4)
        else:
            row["credit_to_width_ratio"] = None

        # Seasonal features
        entry_str = row.get("entry_date")
        if entry_str:
            try:
                dt = datetime.strptime(entry_str, "%Y-%m-%d")
                row["month_of_year"] = dt.month
                row["week_of_year"] = dt.isocalendar()[1]
            except (ValueError, TypeError):
                row["month_of_year"] = None
                row["week_of_year"] = None
        else:
            row["month_of_year"] = None
            row["week_of_year"] = None

        # VIX moving averages + change
        if entry_str and vix_series is not None:
            try:
                entry_ts = pd.Timestamp(entry_str)
                vix_hist = vix_series.loc[vix_series.index <= entry_ts]
                if len(vix_hist) >= 10:
                    row["vix_ma10"] = round(float(vix_hist.tail(10).mean()), 2)
                else:
                    row["vix_ma10"] = None
                if len(vix_hist) >= 5:
                    row["vix_change_5d"] = round(
                        float(vix_hist.iloc[-1]) - float(vix_hist.iloc[-5]), 2
                    )
                else:
                    row["vix_change_5d"] = None
            except Exception:
                row["vix_ma10"] = None
                row["vix_change_5d"] = None
        else:
            row["vix_ma10"] = None
            row["vix_change_5d"] = None

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Dedup Logic
# ═══════════════════════════════════════════════════════════════════════════

def dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate trades and mark exit variants.

    Entry key: entry_date|underlying|strategy_type|spread_type|short_strike|spread_width
    Exit key: exit_date|exit_reason
    Full dedup key = entry_key + exit_key

    - Drop exact duplicates (same entry AND exit)
    - Keep rows with same entry but different exit (mark is_exit_variant=True)
    - Assign entry_group_id for grouping variants
    """
    if df.empty:
        return df

    # Build keys
    entry_cols = ["entry_date", "underlying", "strategy_type", "spread_type",
                  "short_strike", "spread_width"]
    exit_cols = ["exit_date", "exit_reason"]
    full_cols = entry_cols + exit_cols

    # Fill NaN for key building (avoid NaN != NaN issues)
    for col in full_cols:
        if col not in df.columns:
            df[col] = None

    def _make_key(row, cols):
        return "|".join(str(row.get(c, "")) for c in cols)

    df["_entry_key"] = df.apply(lambda r: _make_key(r, entry_cols), axis=1)
    df["_full_key"] = df.apply(lambda r: _make_key(r, full_cols), axis=1)

    raw_count = len(df)

    # Drop exact duplicates (same entry AND exit)
    df = df.drop_duplicates(subset=["_full_key"], keep="first").copy()
    exact_dupes = raw_count - len(df)

    # Assign entry_group_id and mark exit variants
    entry_groups = {}
    group_counter = 0
    group_ids = []
    is_variant = []

    entry_key_counts = df["_entry_key"].value_counts()

    for _, row in df.iterrows():
        ek = row["_entry_key"]
        if ek not in entry_groups:
            entry_groups[ek] = group_counter
            group_counter += 1
        group_ids.append(entry_groups[ek])
        is_variant.append(entry_key_counts[ek] > 1)

    df["entry_group_id"] = group_ids
    df["is_exit_variant"] = is_variant

    # Cleanup internal keys
    df = df.drop(columns=["_entry_key", "_full_key"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    variant_count = df["is_exit_variant"].sum()
    unique_entries = len(entry_groups)

    logger.info(
        "Dedup: %d raw → %d final (%d exact dupes removed, %d unique entries, %d exit variants)",
        raw_count, len(df), exact_dupes, unique_entries, variant_count,
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Quality Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_quality_report(
    df: pd.DataFrame,
    raw_count: int,
    elapsed_sec: float,
    configs_run: int,
    backtests_run: int,
) -> str:
    """Generate markdown quality report."""
    lines = [
        "# Harvest V2 — Quality Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Runtime:** {elapsed_sec:.0f}s ({elapsed_sec/60:.1f} min)",
        f"**Configs run:** {configs_run}",
        f"**Backtests run:** {backtests_run}",
        "",
        "---",
        "",
        "## 1. Dataset Summary",
        "",
        f"- **Total trades:** {len(df)}",
        f"- **Raw before dedup:** {raw_count}",
        f"- **Dedup removed:** {raw_count - len(df)}",
        f"- **Date range:** {df['entry_date'].min()} to {df['entry_date'].max()}",
        f"- **Strategies:** {sorted(df['strategy_type'].unique().tolist())}",
        f"- **Overall win rate:** {df['win'].mean() * 100:.1f}%",
        f"- **Avg return/trade:** {df['return_pct'].mean():.2f}%",
        "",
    ]

    # Exit variant stats
    if "is_exit_variant" in df.columns:
        n_variants = df["is_exit_variant"].sum()
        n_groups = df["entry_group_id"].nunique()
        lines.append(f"- **Unique entry groups:** {n_groups}")
        lines.append(f"- **Exit variants:** {n_variants}")
        lines.append("")

    # Strategy breakdown
    lines.extend(["## 2. Strategy Breakdown", ""])
    lines.append("| Strategy | Trades | Win Rate | Avg Return | Avg Hold Days |")
    lines.append("|----------|--------|----------|------------|---------------|")
    for strat in sorted(df["strategy_type"].unique()):
        s = df[df["strategy_type"] == strat]
        lines.append(
            f"| {strat} | {len(s)} | {s['win'].mean()*100:.1f}% | "
            f"{s['return_pct'].mean():.2f}% | {s['hold_days'].mean():.1f} |"
        )
    lines.append("")

    # Trades by year × regime
    lines.extend(["## 3. Trades by Year × Regime", ""])
    regimes = sorted(df["regime"].dropna().unique())
    header = "| Year | " + " | ".join(regimes) + " | Total |"
    sep = "|------|" + "|".join(["------"] * len(regimes)) + "|-------|"
    lines.append(header)
    lines.append(sep)
    for year in sorted(df["year"].unique()):
        yr_df = df[df["year"] == year]
        counts = [str(len(yr_df[yr_df["regime"] == r])) for r in regimes]
        lines.append(f"| {year} | " + " | ".join(counts) + f" | {len(yr_df)} |")
    lines.append("")

    # Win rate by regime
    lines.extend(["## 4. Win Rate by Regime", ""])
    lines.append("| Regime | Trades | Win Rate | Avg Return |")
    lines.append("|--------|--------|----------|------------|")
    for regime in regimes:
        s = df[df["regime"] == regime]
        lines.append(
            f"| {regime} | {len(s)} | {s['win'].mean()*100:.1f}% | "
            f"{s['return_pct'].mean():.2f}% |"
        )
    lines.append("")

    # Win rate by year
    lines.extend(["## 5. Win Rate by Year", ""])
    lines.append("| Year | Trades | Win Rate | Avg Return | Total PnL |")
    lines.append("|------|--------|----------|------------|-----------|")
    for year in sorted(df["year"].unique()):
        s = df[df["year"] == year]
        lines.append(
            f"| {year} | {len(s)} | {s['win'].mean()*100:.1f}% | "
            f"{s['return_pct'].mean():.2f}% | ${s['pnl'].sum():,.0f} |"
        )
    lines.append("")

    # Feature completeness
    lines.extend(["## 6. Feature Completeness", ""])
    lines.append("| Feature | Non-null | Non-null % |")
    lines.append("|---------|----------|------------|")
    for col in sorted(df.columns):
        nn = df[col].notna().sum()
        pct = nn / len(df) * 100
        lines.append(f"| {col} | {nn} | {pct:.1f}% |")
    lines.append("")

    # Top configs by trade count
    lines.extend(["## 7. Top 10 Configs by Trade Count", ""])
    lines.append("| Config | Trades | Win Rate |")
    lines.append("|--------|--------|----------|")
    top = df["config_label"].value_counts().head(10)
    for cfg, cnt in top.items():
        s = df[df["config_label"] == cfg]
        lines.append(f"| {cfg} | {cnt} | {s['win'].mean()*100:.1f}% |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by compass/harvest_trades_v2.py*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mass Trade Harvester V2")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show config counts without running backtests",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=list(GRID_GENERATORS.keys()),
        choices=list(GRID_GENERATORS.keys()),
        help="Which strategies to harvest (default: all)",
    )
    args = parser.parse_args()

    # Build all configs
    all_configs: List[Tuple[str, str, Dict]] = []  # (label, strategy_name, params)
    for short in args.strategies:
        strategy_name, grid_fn = GRID_GENERATORS[short]
        grid = grid_fn()
        for label, params in grid:
            all_configs.append((label, strategy_name, params))

    total_configs = len(all_configs)
    total_backtests = total_configs * len(YEARS)

    logger.info("=" * 60)
    logger.info("HARVEST V2 — Mass Trade Collection")
    logger.info("=" * 60)
    logger.info("Strategies: %s", args.strategies)
    logger.info("Configs: %d", total_configs)
    logger.info("Years: %s", YEARS)
    logger.info("Total backtests: %d", total_backtests)

    # Per-strategy breakdown
    for short in args.strategies:
        strategy_name, grid_fn = GRID_GENERATORS[short]
        grid = grid_fn()
        logger.info("  %s (%s): %d configs", short.upper(), strategy_name, len(grid))

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN — no backtests executed")
        logger.info("Estimated runtime: %d-%d minutes", total_backtests // 30, total_backtests // 15)
        # Show sample configs
        for short in args.strategies:
            strategy_name, grid_fn = GRID_GENERATORS[short]
            grid = grid_fn()
            logger.info("")
            logger.info("%s sample configs:", short.upper())
            for label, params in grid[:3]:
                key_params = {k: v for k, v in params.items()
                              if k in ("target_dte", "otm_pct", "direction", "rsi_min",
                                       "rsi_max", "min_iv_rank", "mode", "event_types",
                                       "trend_ma_period", "ema_fast", "ema_slow", "min_adx")}
                logger.info("  %s: %s", label, key_params)
            if len(grid) > 3:
                logger.info("  ... and %d more", len(grid) - 3)
        return

    # Full run
    t0 = time.time()

    # Load market data once
    logger.info("")
    logger.info("Loading market data...")
    full_spy, full_vix = _load_full_market_data()
    full_spy_closes = full_spy["Close"]

    all_trades: List[Dict] = []
    configs_run = 0
    backtests_run = 0
    errors = 0

    for idx, (label, strategy_name, params) in enumerate(all_configs):
        configs_run += 1
        config_trades = 0

        logger.info(
            "[%d/%d] %s (%s)",
            idx + 1, total_configs, label, strategy_name,
        )

        for year in YEARS:
            try:
                bt, results = run_single_config(strategy_name, params, year)
                backtests_run += 1

                # Enrich trades
                trades = enrich_trades(
                    bt, year,
                    spy_closes=full_spy_closes,
                    vix_series=full_vix,
                )

                # Extended enrichment
                trades = extend_enrichment(trades, label, full_vix)
                config_trades += len(trades)
                all_trades.extend(trades)

            except Exception as e:
                logger.warning("  ERROR %s year=%d: %s", label, year, e)
                errors += 1
                backtests_run += 1

        logger.info("  → %d trades", config_trades)

    # Build DataFrame
    logger.info("")
    logger.info("Building DataFrame from %d raw trades...", len(all_trades))
    df = pd.DataFrame(all_trades)

    if df.empty:
        logger.error("No trades collected! Check strategy configs.")
        return

    raw_count = len(df)

    # Dedup
    df = dedup_trades(df)

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved %d trades to %s", len(df), OUTPUT_CSV)

    # Generate quality report
    elapsed = time.time() - t0
    report = generate_quality_report(
        df, raw_count, elapsed, configs_run, backtests_run,
    )
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report)
    logger.info("Saved quality report to %s", OUTPUT_REPORT)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("HARVEST COMPLETE")
    logger.info("=" * 60)
    logger.info("Total trades: %d (from %d raw)", len(df), raw_count)
    logger.info("Configs run: %d, Backtests: %d, Errors: %d", configs_run, backtests_run, errors)
    logger.info("Date range: %s to %s", df["entry_date"].min(), df["entry_date"].max())
    logger.info("Strategies: %s", df["strategy_type"].value_counts().to_dict())
    logger.info("Overall win rate: %.1f%%", df["win"].mean() * 100)
    logger.info("Feature columns: %d", len(df.columns))
    logger.info("Elapsed: %.0fs (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
