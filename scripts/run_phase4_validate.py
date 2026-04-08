#!/usr/bin/env python3
"""
run_phase4_validate.py — Phase 4 COMPASS A/B Validation

Runs four backtests from YAML configs:
  1. compass_baseline.yaml  → COMPASS OFF (must reproduce EXP-401)
  2. compass_c001.yaml      → macro sizing ON
  3. compass_c002.yaml      → RRG sector filter ON
  4. compass_c003.yaml      → macro sizing + RRG filter ON

Results saved to: results/phase4_compass_ab_results.json

Hard gates:
  - ROBUST >= 0.90
  - 6/6 years profitable
  - trades >= 250
  - 2022 return >= +5.0%

Usage:
    PYTHONPATH=. python3 scripts/run_phase4_validate.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.portfolio_backtester import PortfolioBacktester
from strategies import STRATEGY_REGISTRY
from strategies.base import MarketSnapshot, Signal

logger = logging.getLogger(__name__)

TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
YEARS = list(range(2020, 2026))

# COMPASS thresholds (from config comments)
MACRO_FEAR_THRESHOLD = 45
MACRO_GREED_THRESHOLD = 75
MACRO_FEAR_BOOST = 1.1
MACRO_GREED_REDUCE = 0.8
RRG_BREADTH_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# COMPASS Data Loader — pre-loads macro scores and RRG breadth from macro_db
# ---------------------------------------------------------------------------

def _load_macro_scores() -> Dict[str, float]:
    """Load weekly macro scores from macro_db, keyed by date string."""
    try:
        from compass.macro_db import get_db
        conn = get_db()
        rows = conn.execute(
            "SELECT date, overall FROM macro_score "
            "WHERE overall IS NOT NULL ORDER BY date"
        ).fetchall()
        conn.close()
        return {r["date"]: r["overall"] for r in rows}
    except Exception as e:
        logger.warning("Failed to load macro scores: %s", e)
        return {}


def _load_rrg_breadth() -> Dict[str, float]:
    """Load weekly RRG breadth (fraction of sectors in Leading/Improving)."""
    try:
        from compass.macro_db import get_db
        conn = get_db()
        rows = conn.execute(
            "SELECT date, rrg_quadrant, COUNT(*) AS n FROM sector_rs "
            "WHERE date >= '2020-01-01' GROUP BY date, rrg_quadrant"
        ).fetchall()
        conn.close()

        # Aggregate per date
        date_totals: Dict[str, int] = {}
        date_positive: Dict[str, int] = {}
        for r in rows:
            d = r["date"]
            date_totals[d] = date_totals.get(d, 0) + r["n"]
            if r["rrg_quadrant"] in ("Leading", "Improving"):
                date_positive[d] = date_positive.get(d, 0) + r["n"]

        result = {}
        for d in sorted(date_totals):
            if date_totals[d] > 0:
                result[d] = date_positive.get(d, 0) / date_totals[d]
        return result
    except Exception as e:
        logger.warning("Failed to load RRG breadth: %s", e)
        return {}


def _forward_fill_to_daily(
    weekly: Dict[str, float], start_year: int = 2020, end_year: int = 2025,
) -> Dict[str, float]:
    """Forward-fill weekly data to daily (YYYY-MM-DD keys)."""
    import pandas as pd

    if not weekly:
        return {}

    dates = sorted(weekly.keys())
    values = [weekly[d] for d in dates]
    idx = pd.to_datetime(dates)
    series = pd.Series(values, index=idx)

    # Reindex to daily
    daily_idx = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31", freq="B"  # business days
    )
    daily = series.reindex(daily_idx).ffill().bfill()
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in daily.items()}


# ---------------------------------------------------------------------------
# CompassBacktester — subclass with COMPASS hooks
# ---------------------------------------------------------------------------

class CompassBacktester(PortfolioBacktester):
    """Portfolio backtester with optional COMPASS extensions."""

    def __init__(
        self,
        *args,
        compass_config: Optional[Dict] = None,
        macro_scores_daily: Optional[Dict[str, float]] = None,
        rrg_breadth_daily: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._compass_cfg = compass_config or {}
        self._macro_scores_daily = macro_scores_daily or {}
        self._rrg_breadth_daily = rrg_breadth_daily or {}
        self._current_date_str: str = ""

        self._macro_sizing_on = self._compass_cfg.get("macro_sizing", False)
        self._rrg_filter_on = self._compass_cfg.get("rrg_filter", False)

        # Stats counters
        self._macro_boosts = 0
        self._macro_reduces = 0
        self._rrg_blocks = 0

    def _build_market_snapshot(self, date_ts, date_dt) -> MarketSnapshot:
        """Override to populate macro_score and rrg_breadth."""
        snapshot = super()._build_market_snapshot(date_ts, date_dt)

        date_str = date_dt.strftime("%Y-%m-%d")
        self._current_date_str = date_str

        if self._macro_sizing_on:
            snapshot.macro_score = self._macro_scores_daily.get(date_str)

        if self._rrg_filter_on:
            snapshot.rrg_breadth = self._rrg_breadth_daily.get(date_str)

        return snapshot

    def _can_accept(self, signal: Signal) -> bool:
        """Override to add RRG breadth filter for bullish entries."""
        if not super()._can_accept(signal):
            return False

        if self._rrg_filter_on:
            # Block bull_put entries when aggregate sector breadth is weak.
            # Credit spread direction is TradeDirection.SHORT; the actual
            # spread type (bull_put / bear_call) lives in signal.metadata.
            spread_type = (signal.metadata or {}).get("spread_type", "")
            if spread_type == "bull_put":
                breadth = self._rrg_breadth_daily.get(self._current_date_str)
                if breadth is not None and breadth < RRG_BREADTH_THRESHOLD:
                    self._rrg_blocks += 1
                    return False

        return True

    def _open_position(self, signal, contracts, date):
        """Override to apply macro-score sizing adjustment."""
        if self._macro_sizing_on:
            macro_score = self._macro_scores_daily.get(
                date.strftime("%Y-%m-%d")
            )
            if macro_score is not None:
                if macro_score < MACRO_FEAR_THRESHOLD:
                    contracts = max(1, int(round(contracts * MACRO_FEAR_BOOST)))
                    self._macro_boosts += 1
                elif macro_score > MACRO_GREED_THRESHOLD:
                    contracts = max(1, int(round(contracts * MACRO_GREED_REDUCE)))
                    self._macro_reduces += 1

        return super()._open_position(signal, contracts, date)

    def get_compass_stats(self) -> Dict[str, int]:
        return {
            "macro_boosts": self._macro_boosts,
            "macro_reduces": self._macro_reduces,
            "rrg_blocks": self._rrg_blocks,
        }


# ---------------------------------------------------------------------------
# YAML Config Parser
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> Dict:
    """Load and parse a COMPASS YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_strategy_params(config: Dict) -> Tuple[Dict, Dict]:
    """Extract CS and SS strategy params from YAML config.

    Returns (cs_params, ss_params).
    """
    strategy = config.get("strategy", {})
    risk = config.get("risk", {})
    technical = strategy.get("technical", {})

    # Credit Spread params — aligned with champion config structure
    cs_params = {
        "direction": strategy.get("direction", "regime_adaptive"),
        "trend_ma_period": technical.get("slow_ma", 80),
        "target_dte": strategy.get("target_dte", 15),
        "min_dte": strategy.get("min_dte", 15),
        "otm_pct": strategy.get("otm_pct", 0.02),
        "spread_width": strategy.get("spread_width", 12.0),
        "profit_target_pct": risk.get("profit_target", 55) / 100.0,
        "stop_loss_multiplier": risk.get("stop_loss_multiplier", 1.25),
        "momentum_filter_pct": risk.get("momentum_filter_pct", 2.0),
        "scan_weekday": "any",
        "max_risk_pct": risk.get("max_risk_per_trade", 12.0) / 100.0,
        # Regime scales
        "regime_scale_bull": risk.get("regime_scale_bull", 1.0),
        "regime_scale_bear": risk.get("regime_scale_bear", 0.3),
        "regime_scale_high_vol": risk.get("regime_scale_high_vol", 0.3),
        "regime_scale_low_vol": risk.get("regime_scale_low_vol", 0.8),
        "regime_scale_crash": risk.get("regime_scale_crash", 0.0),
        # Regime config
        "regime_mode": strategy.get("regime_mode", "combo"),
        "regime_config": strategy.get("regime_config", {}),
    }

    # Straddle/Strangle params
    ss_cfg = strategy.get("straddle_strangle", {})
    ss_params = {
        "mode": ss_cfg.get("mode", "short_post_event"),
        "target_dte": ss_cfg.get("target_dte", 5),
        "otm_pct": ss_cfg.get("otm_pct", 0.04),
        "event_iv_boost": ss_cfg.get("event_iv_boost", 0.15),
        "iv_crush_pct": ss_cfg.get("iv_crush_pct", 0.5),
        "profit_target_pct": ss_cfg.get("profit_target_pct", 0.55),
        "stop_loss_pct": ss_cfg.get("stop_loss_pct", 0.45),
        "max_risk_pct": risk.get("straddle_strangle_risk_pct", 3.0) / 100.0,
        "event_types": ss_cfg.get("event_types", "fomc_cpi"),
        # SS Regime scales
        "regime_scale_bull": risk.get("ss_regime_scale_bull", 1.5),
        "regime_scale_bear": risk.get("ss_regime_scale_bear", 1.5),
        "regime_scale_high_vol": risk.get("ss_regime_scale_high_vol", 2.5),
        "regime_scale_low_vol": risk.get("ss_regime_scale_low_vol", 1.0),
        "regime_scale_crash": risk.get("ss_regime_scale_crash", 0.5),
    }

    return cs_params, ss_params


# ---------------------------------------------------------------------------
# Backtest Runner
# ---------------------------------------------------------------------------

def run_single_year(
    cs_params: Dict,
    ss_params: Dict,
    year: int,
    compass_config: Dict,
    macro_daily: Dict[str, float],
    rrg_daily: Dict[str, float],
    max_pos: int = 12,
    max_per_strat: int = 5,
) -> Dict:
    """Run one year of the CS+SS blend through CompassBacktester."""
    cs_cls = STRATEGY_REGISTRY["credit_spread"]
    ss_cls = STRATEGY_REGISTRY["straddle_strangle"]

    cs_inst = cs_cls(dict(cs_params))
    ss_inst = ss_cls(dict(ss_params))

    bt = CompassBacktester(
        strategies=[("credit_spread", cs_inst), ("straddle_strangle", ss_inst)],
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=max_pos,
        max_positions_per_strategy=max_per_strat,
        compass_config=compass_config,
        macro_scores_daily=macro_daily,
        rrg_breadth_daily=rrg_daily,
    )

    raw = bt.run()
    combined = raw.get("combined", raw)

    return {
        "return_pct": combined.get("return_pct", 0),
        "max_drawdown": combined.get("max_drawdown", 0),
        "total_trades": combined.get("total_trades", 0),
        "win_rate": combined.get("win_rate", 0),
        "sharpe_ratio": combined.get("sharpe_ratio", 0),
        "compass_stats": bt.get_compass_stats(),
    }


def run_experiment(
    label: str,
    config_path: Path,
    macro_daily: Dict[str, float],
    rrg_daily: Dict[str, float],
) -> Dict:
    """Run a full 6-year experiment from a YAML config."""
    config = load_yaml_config(config_path)
    cs_params, ss_params = build_strategy_params(config)
    compass_cfg = config.get("compass", {})
    experiment_id = config.get("experiment_id", label)

    print(f"\n{'='*70}")
    print(f"  {experiment_id}: {label}")
    print(f"  macro_sizing={compass_cfg.get('macro_sizing', False)} "
          f"rrg_filter={compass_cfg.get('rrg_filter', False)}")
    print(f"{'='*70}")

    yearly: Dict[str, Dict] = {}
    total_compass_stats = {"macro_boosts": 0, "macro_reduces": 0, "rrg_blocks": 0}

    for year in YEARS:
        t0 = time.time()
        print(f"  {year}...", end=" ", flush=True)
        try:
            r = run_single_year(
                cs_params, ss_params, year, compass_cfg, macro_daily, rrg_daily,
            )
            elapsed = time.time() - t0
            print(f"{r['return_pct']:+.1f}%  {r['total_trades']} trades  "
                  f"DD={r['max_drawdown']:.1f}%  ({elapsed:.0f}s)"
                  f"  compass={r['compass_stats']}")
            yearly[str(year)] = r

            # Accumulate compass stats
            for k, v in r["compass_stats"].items():
                total_compass_stats[k] += v
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.0f}s): {e}")
            import traceback
            traceback.print_exc()
            yearly[str(year)] = {
                "return_pct": 0, "total_trades": 0, "max_drawdown": 0,
                "win_rate": 0, "sharpe_ratio": 0, "error": str(e),
                "compass_stats": {},
            }

    # Summary
    rets = [yearly[str(y)]["return_pct"] for y in YEARS]
    dds = [yearly[str(y)]["max_drawdown"] for y in YEARS]
    trades = [yearly[str(y)]["total_trades"] for y in YEARS]

    summary = {
        "experiment_id": experiment_id,
        "config_path": str(config_path.name),
        "compass": compass_cfg,
        "yearly": yearly,
        "avg_return": round(sum(rets) / len(rets), 2),
        "min_return": round(min(rets), 2),
        "max_return": round(max(rets), 2),
        "worst_dd": round(min(dds), 2),
        "total_trades": sum(trades),
        "years_profitable": sum(1 for r in rets if r > 0),
        "year_2022_return": yearly.get("2022", {}).get("return_pct", 0),
        "compass_stats": total_compass_stats,
    }

    print(f"\n  SUMMARY: avg={summary['avg_return']:+.1f}%  "
          f"worst_dd={summary['worst_dd']:.1f}%  "
          f"trades={summary['total_trades']}  "
          f"yrs_profitable={summary['years_profitable']}/6  "
          f"2022={summary['year_2022_return']:+.1f}%")

    return summary


# ---------------------------------------------------------------------------
# Gate Evaluation
# ---------------------------------------------------------------------------

def evaluate_gates(result: Dict) -> Dict:
    """Evaluate hard gates for a backtest result.

    ROBUST score formula (aligned with validate_params.py composite):
      A (consistency)   × 0.30 — years profitable / 6
      B (return quality) × 0.25 — min(1.0, avg_return / 30)
      C (drawdown)       × 0.20 — max(0, 1 - |worst_dd| / 30)
      D (trade count)    × 0.15 — min(1.0, total_trades / 250)
      E (worst year)     × 0.10 — 1.0 if min_year > 5%, 0.5 if > 0, else 0

    Calibrated: EXP-401 baseline scores 0.953 (original validate_params: 0.951).
    """
    avg_ret = result["avg_return"]
    worst_dd = result["worst_dd"]
    yrs_profit = result["years_profitable"]
    total_trades = result["total_trades"]
    year_2022 = result["year_2022_return"]
    min_return = result["min_return"]

    a_score = yrs_profit / 6.0
    b_score = min(1.0, max(0, avg_ret / 30.0))
    c_score = max(0, 1.0 - abs(worst_dd) / 30.0)
    d_score = min(1.0, total_trades / 250.0)
    e_score = 1.0 if min_return >= 5.0 else (0.5 if min_return > 0 else 0.0)

    robust_score = (
        a_score * 0.30
        + b_score * 0.25
        + c_score * 0.20
        + d_score * 0.15
        + e_score * 0.10
    )
    robust_score = round(robust_score, 3)

    gates = {
        "robust_score": robust_score,
        "robust_components": {
            "A_consistency": round(a_score, 3),
            "B_return_quality": round(b_score, 3),
            "C_drawdown": round(c_score, 3),
            "D_trade_count": round(d_score, 3),
            "E_worst_year": round(e_score, 3),
        },
        "robust_pass": robust_score >= 0.90,
        "profitable_6_6": yrs_profit == 6,
        "trades_pass": total_trades >= 250,
        "year_2022_pass": year_2022 >= 5.0,
        "all_pass": (
            robust_score >= 0.90
            and yrs_profit == 6
            and total_trades >= 250
            and year_2022 >= 5.0
        ),
    }

    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.WARNING)

    configs_dir = ROOT / "configs"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    config_files = [
        ("BASELINE (EXP-401 regression)", "compass_baseline.yaml"),
        ("C001 (macro sizing)", "compass_c001.yaml"),
        ("C002 (RRG filter)", "compass_c002.yaml"),
        ("C003 (macro + RRG)", "compass_c003.yaml"),
    ]

    # Validate all configs exist
    for label, fname in config_files:
        path = configs_dir / fname
        if not path.exists():
            print(f"ERROR: Config not found: {path}")
            sys.exit(1)

    # Pre-load COMPASS data
    print("Loading COMPASS macro/RRG data from macro_db...")
    t0 = time.time()
    macro_weekly = _load_macro_scores()
    rrg_weekly = _load_rrg_breadth()
    macro_daily = _forward_fill_to_daily(macro_weekly)
    rrg_daily = _forward_fill_to_daily(rrg_weekly)
    print(f"  Loaded {len(macro_weekly)} weekly macro scores → {len(macro_daily)} daily")
    print(f"  Loaded {len(rrg_weekly)} weekly RRG breadths → {len(rrg_daily)} daily")
    print(f"  ({time.time()-t0:.1f}s)")

    # Run all experiments
    all_results: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "tickers": TICKERS,
        "starting_capital": STARTING_CAPITAL,
        "years": YEARS,
        "experiments": {},
    }

    baseline_result = None
    for label, fname in config_files:
        path = configs_dir / fname
        result = run_experiment(label, path, macro_daily, rrg_daily)
        gates = evaluate_gates(result)
        result["gates"] = gates

        key = fname.replace(".yaml", "").replace("compass_", "")
        all_results["experiments"][key] = result

        if "baseline" in key:
            baseline_result = result

        # Print gate evaluation
        print(f"\n  GATES:")
        print(f"    ROBUST score: {gates['robust_score']:.3f} "
              f"({'PASS' if gates['robust_pass'] else 'FAIL'} >= 0.90)")
        print(f"    6/6 profitable: {'PASS' if gates['profitable_6_6'] else 'FAIL'}")
        print(f"    Trades >= 250: {result['total_trades']} "
              f"({'PASS' if gates['trades_pass'] else 'FAIL'})")
        print(f"    2022 >= +5.0%: {result['year_2022_return']:+.1f}% "
              f"({'PASS' if gates['year_2022_pass'] else 'FAIL'})")
        print(f"    ALL GATES: {'PASS' if gates['all_pass'] else 'FAIL'}")

    # A/B comparison vs baseline
    if baseline_result:
        print(f"\n{'='*70}")
        print("  A/B COMPARISON vs BASELINE")
        print(f"{'='*70}")

        for key, result in all_results["experiments"].items():
            if key == "baseline":
                continue
            delta_ret = result["avg_return"] - baseline_result["avg_return"]
            delta_dd = result["worst_dd"] - baseline_result["worst_dd"]
            delta_trades = result["total_trades"] - baseline_result["total_trades"]

            result["vs_baseline"] = {
                "delta_avg_return": round(delta_ret, 2),
                "delta_worst_dd": round(delta_dd, 2),
                "delta_total_trades": delta_trades,
            }

            print(f"\n  {key}:")
            print(f"    avg_return: {result['avg_return']:+.1f}% "
                  f"(delta: {'+' if delta_ret >= 0 else ''}{delta_ret:.2f}pp)")
            print(f"    worst_dd:   {result['worst_dd']:.1f}% "
                  f"(delta: {'+' if delta_dd >= 0 else ''}{delta_dd:.2f}pp)")
            print(f"    trades:     {result['total_trades']} "
                  f"(delta: {'+' if delta_trades >= 0 else ''}{delta_trades})")

    # Regression check for baseline
    if baseline_result:
        print(f"\n{'='*70}")
        print("  EXP-401 REGRESSION CHECK")
        print(f"{'='*70}")

        exp401_targets = {
            "avg_return": 40.7,
            "worst_dd": -7.0,
            "years_profitable": 6,
        }

        for metric, target in exp401_targets.items():
            actual = baseline_result[metric]
            tol = 0.1 if "dd" in metric else 0.1
            if isinstance(target, int):
                status = "PASS" if actual == target else "FAIL"
            else:
                status = "PASS" if abs(actual - target) <= tol else "FAIL"
            print(f"  {metric}: target={target} actual={actual} → {status}")

    # Save results
    output_path = results_dir / "phase4_compass_ab_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
