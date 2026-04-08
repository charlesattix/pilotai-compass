"""
compass/experiment_runner.py — Automated experiment pipeline.

Run a complete backtest experiment with one command:
  1. Takes an ExperimentSpec (strategy, params, ticker, dates, validation)
  2. Runs backtest with walk-forward validation
  3. Computes North Star metrics (CAGR, DD, Sharpe, capacity, SPY corr)
  4. Generates standardized HTML report
  5. Updates REGISTRY.md with results
  6. Flags pass/fail vs North Star filters

Usage:
    from compass.experiment_runner import ExperimentRunner, ExperimentSpec

    spec = ExperimentSpec(
        experiment_id="EXP-1700",
        name="My Strategy",
        strategy_type="credit_spread",
        ticker="SPY",
        params={...},
    )
    runner = ExperimentRunner()
    result = runner.run(spec)
    runner.generate_report(result, "reports/exp1700.html")
    runner.update_registry(result)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# North Star thresholds
# ═══════════════════════════════════════════════════════════════════════════

NORTH_STAR = {
    "min_cagr": 0.55,           # 55% annual return
    "min_sharpe": 3.0,          # risk-adjusted
    "max_dd": 0.30,             # 30% max drawdown
    "min_profitable_years": 4,  # out of 6
    "min_trades": 15,           # statistical significance
    "max_spy_corr": 0.50,       # market neutrality
    "min_oos_sharpe": 0.5,      # walk-forward passes
    "min_win_rate": 0.50,       # above coin-flip
}


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentSpec:
    """Everything needed to run one experiment."""
    experiment_id: str              # e.g. "EXP-1700"
    name: str                       # human-readable
    strategy_type: str              # "credit_spread", "iron_condor", "pairs", "custom"
    ticker: str                     # "SPY", "GLD", etc.
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    capital: float = 100_000
    params: Dict[str, Any] = field(default_factory=dict)
    # Backtester config overrides
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    # Validation
    validation: str = "walk_forward"  # "walk_forward", "cpcv", "none"
    oos_start_year: int = 2022
    # Custom backtest function (for non-standard strategies)
    custom_runner: Optional[Any] = None  # callable(spec) -> Dict
    description: str = ""
    data_source: str = "ironvault"   # "ironvault" or "synthetic"
    # Documentation fields
    hypothesis: str = ""            # what we expect and why
    success_criteria: Dict[str, Any] = field(default_factory=dict)  # e.g. {"min_oos_sharpe": 1.0}


@dataclass
class NorthStarCheck:
    """Pass/fail for each North Star metric."""
    name: str
    target: str
    actual: str
    passed: bool


@dataclass
class WalkForwardWindow:
    """One walk-forward validation window."""
    is_period: str
    oos_period: str
    is_trades: int
    oos_trades: int
    is_sharpe: float
    oos_sharpe: float
    oos_pnl: float
    oos_wr: float


@dataclass
class ExperimentResult:
    """Complete output of an experiment run."""
    spec: ExperimentSpec
    # Core metrics
    n_trades: int = 0
    total_pnl: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    spy_corr: float = 0.0
    avg_hold_days: float = 0.0
    # Walk-forward
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    wf_ratio: float = 0.0
    wf_windows: List[WalkForwardWindow] = field(default_factory=list)
    # Per-year
    yearly: Dict[int, Dict] = field(default_factory=dict)
    profitable_years: int = 0
    total_years: int = 0
    # Regime breakdown
    regime_stats: Dict[str, Dict] = field(default_factory=dict)
    # Capacity
    estimated_capacity: str = "unknown"
    # North Star
    north_star_checks: List[NorthStarCheck] = field(default_factory=list)
    north_star_passed: int = 0
    north_star_total: int = 0
    tier: int = 4  # 1=production, 2=promising, 3=marginal, 4=dead
    verdict: str = ""
    # Raw data
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List = field(default_factory=list)
    # Timing
    run_time_seconds: float = 0.0
    timestamp: str = ""
    data_source: str = "ironvault"
    errors: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sharpe(pnls: np.ndarray) -> float:
    if len(pnls) < 2:
        return 0.0
    s = np.std(pnls, ddof=1)
    return float(np.mean(pnls) / s * math.sqrt(min(len(pnls), 252))) if s > 1e-9 else 0.0


def _cagr(total_pnl: float, capital: float, years: float) -> float:
    if years <= 0 or total_pnl <= -capital:
        return -1.0
    return ((1 + total_pnl / capital) ** (1 / max(years, 0.5))) - 1


def _max_dd(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    return float(dd.max())


def _spy_correlation(trade_pnls: Dict[str, float], spy_df: pd.DataFrame) -> float:
    """Correlate trade returns with SPY daily returns."""
    if not trade_pnls or spy_df.empty:
        return 0.0
    spy_ret = spy_df["Close"].pct_change().fillna(0)
    ts = pd.Series(trade_pnls)
    ts.index = pd.to_datetime(ts.index)
    ci = ts.index.intersection(spy_ret.index)
    if len(ci) < 5:
        return 0.0
    return float(np.corrcoef(
        ts.reindex(ci).fillna(0), spy_ret.reindex(ci).fillna(0)
    )[0, 1])


def _load_spy() -> pd.DataFrame:
    from backtest.backtester import _yf_download_safe
    df = _yf_download_safe("SPY", "2019-06-01", "2027-01-01")
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Default config builder
# ═══════════════════════════════════════════════════════════════════════════

def build_default_config(spec: ExperimentSpec) -> Dict:
    """Build a Backtester-compatible config from an ExperimentSpec."""
    params = spec.params

    config = {
        "strategy": {
            "direction": params.get("direction", "both"),
            "regime_mode": params.get("regime_mode", "combo"),
            "regime_config": params.get("regime_config", {
                "signals": ["price_vs_ma200", "rsi_momentum", "vix_structure"],
                "ma_slow_period": 200,
            }),
            "min_dte": params.get("min_dte", 15),
            "target_dte": params.get("target_dte", 30),
            "otm_pct": params.get("otm_pct", 0.05),
            "spread_width": params.get("spread_width", 5.0),
            "profit_target_pct": params.get("profit_target_pct", 0.50),
            "stop_loss_multiplier": params.get("stop_loss_multiplier", 2.0),
            "max_risk_pct": params.get("max_risk_pct", 0.05),
            "iron_condor": params.get("iron_condor", {"enabled": False}),
        },
        "risk": {
            "max_risk_per_trade": params.get("max_risk_pct", 5.0) * 100
                if params.get("max_risk_pct", 0.05) < 1 else params.get("max_risk_pct", 5.0),
            "sizing_mode": params.get("sizing_mode", "flat"),
            "max_contracts": params.get("max_contracts", 25),
            "max_positions": params.get("max_positions", 10),
            "profit_target": params.get("profit_target_pct", 50),
            "stop_loss_multiplier": params.get("stop_loss_multiplier", 2.0),
        },
        "backtest": {
            "starting_capital": spec.capital,
            "commission_per_contract": params.get("commission", 0.65),
            "slippage": params.get("entry_slippage", 0.05),
            "exit_slippage": params.get("exit_slippage", 0.10),
        },
    }

    # Apply config overrides
    for section, overrides in spec.config_overrides.items():
        if section in config and isinstance(overrides, dict):
            config[section].update(overrides)

    return config


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward validation
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_validate(
    trades: List[Dict],
    oos_start_year: int = 2022,
) -> tuple:
    """Walk-forward validation on trade list.

    Returns: (is_sharpe, oos_sharpe, wf_ratio, windows)
    """
    if not trades:
        return 0.0, 0.0, 0.0, []

    df = pd.DataFrame(trades)
    if "exit_date" in df.columns:
        df["date"] = pd.to_datetime(df["exit_date"])
    elif "entry_date" in df.columns:
        df["date"] = pd.to_datetime(df["entry_date"])
    else:
        return 0.0, 0.0, 0.0, []

    df["year"] = df["date"].dt.year
    pnls = df["pnl"].values
    years = sorted(df["year"].unique())

    # Overall IS/OOS split
    is_mask = df["year"] < oos_start_year
    oos_mask = df["year"] >= oos_start_year
    is_sh = _sharpe(pnls[is_mask]) if is_mask.sum() > 1 else 0.0
    oos_sh = _sharpe(pnls[oos_mask]) if oos_mask.sum() > 1 else 0.0
    wf = oos_sh / is_sh if abs(is_sh) > 0.01 else 0.0

    # Rolling 1yr IS / 1yr OOS windows
    windows = []
    for i in range(len(years) - 1):
        is_yr, oos_yr = years[i], years[i + 1]
        is_t = df[df["year"] == is_yr]
        oos_t = df[df["year"] == oos_yr]
        if len(is_t) < 2 or len(oos_t) < 2:
            continue
        is_s = _sharpe(is_t["pnl"].values)
        oos_s = _sharpe(oos_t["pnl"].values)
        oos_pnl = float(oos_t["pnl"].sum())
        oos_wr = float((oos_t["pnl"] > 0).sum()) / len(oos_t)
        windows.append(WalkForwardWindow(
            is_period=str(is_yr), oos_period=str(oos_yr),
            is_trades=len(is_t), oos_trades=len(oos_t),
            is_sharpe=round(is_s, 3), oos_sharpe=round(oos_s, 3),
            oos_pnl=round(oos_pnl, 2), oos_wr=round(oos_wr, 3),
        ))

    return round(is_sh, 3), round(oos_sh, 3), round(wf, 3), windows


# ═══════════════════════════════════════════════════════════════════════════
# North Star evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_north_star(result: ExperimentResult) -> None:
    """Evaluate experiment against North Star thresholds. Mutates result."""
    checks = []
    ns = NORTH_STAR

    def _check(name, target_str, actual_val, actual_str, threshold, higher_is_better=True):
        if higher_is_better:
            passed = actual_val >= threshold
        else:
            passed = actual_val <= threshold
        checks.append(NorthStarCheck(name=name, target=target_str,
                                     actual=actual_str, passed=passed))

    _check("CAGR", f">={ns['min_cagr']:.0%}", result.cagr,
           f"{result.cagr:.1%}", ns["min_cagr"])
    _check("Sharpe", f">={ns['min_sharpe']:.1f}", result.sharpe,
           f"{result.sharpe:.2f}", ns["min_sharpe"])
    _check("Max DD", f"<={ns['max_dd']:.0%}", result.max_dd,
           f"{result.max_dd:.1%}", ns["max_dd"], higher_is_better=False)
    _check("Profitable Years", f">={ns['min_profitable_years']}",
           result.profitable_years, f"{result.profitable_years}/{result.total_years}",
           ns["min_profitable_years"])
    _check("Trade Count", f">={ns['min_trades']}", result.n_trades,
           str(result.n_trades), ns["min_trades"])
    _check("SPY Correlation", f"<={ns['max_spy_corr']:.2f}", abs(result.spy_corr),
           f"{result.spy_corr:.3f}", ns["max_spy_corr"], higher_is_better=False)
    _check("OOS Sharpe", f">={ns['min_oos_sharpe']:.1f}", result.oos_sharpe,
           f"{result.oos_sharpe:.2f}", ns["min_oos_sharpe"])
    _check("Win Rate", f">={ns['min_win_rate']:.0%}", result.win_rate,
           f"{result.win_rate:.1%}", ns["min_win_rate"])

    result.north_star_checks = checks
    result.north_star_passed = sum(1 for c in checks if c.passed)
    result.north_star_total = len(checks)

    # Tier assignment
    passed = result.north_star_passed
    total = result.north_star_total
    if passed == total:
        result.tier = 1
        result.verdict = f"TIER 1 — Production ready ({passed}/{total} North Star)"
    elif passed >= total - 2 and result.oos_sharpe > 0.5:
        result.tier = 2
        result.verdict = f"TIER 2 — Promising ({passed}/{total} North Star)"
    elif passed >= total // 2 and result.n_trades >= 10:
        result.tier = 3
        result.verdict = f"TIER 3 — Marginal ({passed}/{total} North Star)"
    else:
        result.tier = 4
        result.verdict = f"TIER 4 — Dead ({passed}/{total} North Star)"


# ═══════════════════════════════════════════════════════════════════════════
# Capacity estimation (simplified from capacity_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════

_ATM_ADV = {"SPY": 500_000, "GLD": 5_000, "TLT": 8_000, "XLI": 3_000,
            "XLF": 10_000, "QQQ": 50_000, "IBIT": 500}

def estimate_capacity(ticker: str, avg_contracts: int = 5) -> str:
    """Quick capacity estimate based on ticker liquidity."""
    adv = _ATM_ADV.get(ticker, 5_000)
    # At 2% participation rate, max contracts per trade
    max_cts = int(adv * 0.02)
    scale = max_cts / max(avg_contracts, 1)
    max_aum = 100_000 * scale
    if max_aum >= 1e9:
        return f"${max_aum/1e9:.1f}B"
    elif max_aum >= 1e6:
        return f"${max_aum/1e6:.0f}M"
    return f"${max_aum/1e3:.0f}K"


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════

def _check_success_criteria(criteria: Dict[str, Any], result: ExperimentResult) -> None:
    """Check custom success criteria from the spec. Appends to verdict."""
    passed = []
    failed = []
    metric_map = {
        "min_sharpe": ("sharpe", True),
        "min_oos_sharpe": ("oos_sharpe", True),
        "min_cagr": ("cagr", True),
        "max_dd": ("max_dd", False),
        "min_trades": ("n_trades", True),
        "min_win_rate": ("win_rate", True),
        "max_spy_corr": ("spy_corr", False),
    }
    for key, threshold in criteria.items():
        if key in metric_map:
            attr, higher_is_better = metric_map[key]
            val = getattr(result, attr, 0)
            if attr == "spy_corr":
                val = abs(val)
            if higher_is_better:
                ok = val >= threshold
            else:
                ok = val <= threshold
            (passed if ok else failed).append(f"{key}: {val:.3f} vs {threshold}")

    if failed:
        result.verdict += f" | Custom criteria: {len(passed)}/{len(passed)+len(failed)} passed"


class ExperimentRunner:
    """Orchestrates end-to-end experiment execution."""

    def __init__(self):
        self._spy_df = None

    @property
    def spy_df(self) -> pd.DataFrame:
        if self._spy_df is None:
            self._spy_df = _load_spy()
        return self._spy_df

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run a complete experiment from spec to scored result."""
        t0 = datetime.utcnow()
        result = ExperimentResult(
            spec=spec,
            timestamp=t0.isoformat(),
            data_source=spec.data_source,
        )

        try:
            # Custom runner path
            if spec.custom_runner is not None:
                raw = spec.custom_runner(spec)
                self._extract_results(raw, result)
            else:
                raw = self._run_backtester(spec)
                self._extract_backtester_results(raw, result, spec)

            # Walk-forward validation
            if spec.validation != "none" and result.trades:
                is_sh, oos_sh, wf, windows = walk_forward_validate(
                    result.trades, spec.oos_start_year
                )
                result.is_sharpe = is_sh
                result.oos_sharpe = oos_sh
                result.wf_ratio = wf
                result.wf_windows = windows

            # SPY correlation
            if result.trades:
                trade_pnls = {}
                for t in result.trades:
                    d = str(t.get("exit_date", t.get("entry_date", "")))[:10]
                    trade_pnls[d] = trade_pnls.get(d, 0) + t["pnl"]
                result.spy_corr = _spy_correlation(trade_pnls, self.spy_df)

            # Capacity
            avg_cts = spec.params.get("max_contracts", 5)
            result.estimated_capacity = estimate_capacity(spec.ticker, avg_cts)

            # North Star evaluation
            evaluate_north_star(result)

            # Check custom success criteria
            if spec.success_criteria:
                _check_success_criteria(spec.success_criteria, result)

        except Exception as e:
            result.errors.append(f"{type(e).__name__}: {e}")
            result.verdict = f"ERROR: {e}"
            logger.exception("Experiment %s failed", spec.experiment_id)

        result.run_time_seconds = (datetime.utcnow() - t0).total_seconds()
        return result

    def _run_backtester(self, spec: ExperimentSpec) -> Dict:
        """Run using the standard Backtester."""
        from backtest.backtester import Backtester
        from shared.iron_vault import IronVault

        config = build_default_config(spec)

        hd = None
        if spec.data_source == "ironvault":
            hd = IronVault.instance()

        bt = Backtester(config, historical_data=hd,
                        otm_pct=spec.params.get("otm_pct", 0.05))
        return bt.run_backtest(
            ticker=spec.ticker,
            start_date=datetime.strptime(spec.start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(spec.end_date, "%Y-%m-%d"),
        )

    def _extract_backtester_results(self, raw: Dict, result: ExperimentResult,
                                     spec: ExperimentSpec) -> None:
        """Extract metrics from Backtester return dict into ExperimentResult."""
        result.n_trades = raw.get("total_trades", 0)
        result.total_pnl = raw.get("total_pnl", 0)
        result.win_rate = raw.get("win_rate", 0) / 100.0  # Backtester returns %
        result.max_dd = abs(raw.get("max_drawdown", 0)) / 100.0  # % to fraction
        result.sharpe = raw.get("sharpe_ratio", 0)
        result.profit_factor = raw.get("profit_factor", 0)
        result.trades = raw.get("trades", [])
        result.equity_curve = raw.get("equity_curve", [])

        # Compute CAGR
        start = datetime.strptime(spec.start_date, "%Y-%m-%d")
        end = datetime.strptime(spec.end_date, "%Y-%m-%d")
        years = (end - start).days / 365.25
        result.cagr = _cagr(result.total_pnl, spec.capital, years)

        # Yearly breakdown
        if result.trades:
            df = pd.DataFrame(result.trades)
            date_col = "exit_date" if "exit_date" in df.columns else "entry_date"
            if date_col in df.columns:
                df["year"] = pd.to_datetime(df[date_col]).dt.year
                for yr, grp in df.groupby("year"):
                    yp = grp["pnl"].values
                    yn = len(yp)
                    result.yearly[int(yr)] = {
                        "n": yn,
                        "pnl": round(float(yp.sum()), 2),
                        "wr": round(float((yp > 0).sum()) / yn, 3) if yn > 0 else 0,
                        "sharpe": round(_sharpe(yp), 3),
                    }
                result.profitable_years = sum(
                    1 for y in result.yearly.values() if y["pnl"] > 0
                )
                result.total_years = len(result.yearly)

    def _extract_results(self, raw: Dict, result: ExperimentResult) -> None:
        """Extract from a custom runner return dict."""
        result.n_trades = raw.get("n_trades", 0)
        result.total_pnl = raw.get("total_pnl", 0)
        result.cagr = raw.get("cagr", 0)
        result.sharpe = raw.get("sharpe", 0)
        result.max_dd = raw.get("max_dd", 0)
        result.win_rate = raw.get("win_rate", 0)
        result.profit_factor = raw.get("profit_factor", 0)
        result.trades = raw.get("trades", [])
        result.yearly = raw.get("yearly", {})
        result.regime_stats = raw.get("regime_stats", {})
        result.profitable_years = sum(
            1 for y in result.yearly.values()
            if (y.get("pnl", 0) if isinstance(y, dict) else 0) > 0
        )
        result.total_years = len(result.yearly) if result.yearly else 0

    # ── Report generation ──────────────────────────────────────────────

    def generate_report(self, result: ExperimentResult,
                        output_path: str | Path) -> Path:
        """Generate standardized HTML report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def save_json(self, result: ExperimentResult,
                  output_path: str | Path) -> Path:
        """Save result as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experiment_id": result.spec.experiment_id,
            "name": result.spec.name,
            "strategy_type": result.spec.strategy_type,
            "ticker": result.spec.ticker,
            "data_source": result.data_source,
            "timestamp": result.timestamp,
            "run_time_seconds": result.run_time_seconds,
            "n_trades": result.n_trades,
            "total_pnl": result.total_pnl,
            "cagr": result.cagr,
            "sharpe": result.sharpe,
            "max_dd": result.max_dd,
            "win_rate": result.win_rate,
            "spy_corr": result.spy_corr,
            "is_sharpe": result.is_sharpe,
            "oos_sharpe": result.oos_sharpe,
            "wf_ratio": result.wf_ratio,
            "tier": result.tier,
            "verdict": result.verdict,
            "north_star_passed": result.north_star_passed,
            "north_star_total": result.north_star_total,
            "estimated_capacity": result.estimated_capacity,
            "yearly": result.yearly,
            "errors": result.errors,
        }
        output_path.write_text(json.dumps(data, indent=2, default=str))
        return output_path

    # ── Registry update ────────────────────────────────────────────────

    def update_registry(self, result: ExperimentResult,
                        registry_path: Optional[Path] = None) -> None:
        """Append experiment result to REGISTRY.md."""
        if registry_path is None:
            registry_path = ROOT / "REGISTRY.md"
        if not registry_path.exists():
            logger.warning("REGISTRY.md not found at %s", registry_path)
            return

        content = registry_path.read_text(encoding="utf-8")

        # Build the registry line
        status_map = {1: "**LIVE-READY**", 2: "PROMISING", 3: "MARGINAL", 4: "**DEAD**"}
        status = status_map.get(result.tier, "UNKNOWN")
        data_label = "Real" if result.data_source == "ironvault" else "Synth"

        line = (
            f"| {result.spec.experiment_id} | {result.spec.name} | "
            f"{result.spec.strategy_type} | {result.spec.ticker} | "
            f"{status} | {data_label} | "
            f"{result.sharpe:.2f} | {result.cagr:.1%} | {result.max_dd:.1%} | "
            f"{result.spy_corr:.3f} | {result.estimated_capacity} | "
            f"{result.n_trades} | "
            f"{result.verdict} |"
        )

        # Check if experiment already in registry
        if result.spec.experiment_id in content:
            logger.info("Experiment %s already in REGISTRY.md — skipping update",
                        result.spec.experiment_id)
            return

        # Append to the end of the file
        update_block = (
            f"\n\n<!-- Auto-added by experiment_runner {result.timestamp} -->\n"
            f"### {result.spec.experiment_id}: {result.spec.name}\n\n"
            f"| ID | Name | Type | Ticker | Status | Data | Sharpe | CAGR | Max DD | "
            f"SPY Corr | Capacity | Trades | Verdict |\n"
            f"|-----|------|------|--------|--------|------|--------|------|--------|"
            f"----------|----------|--------|--------|\n"
            f"{line}\n"
        )

        registry_path.write_text(content + update_block, encoding="utf-8")
        logger.info("Updated REGISTRY.md with %s", result.spec.experiment_id)


# ═══════════════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════════════


class BatchRunner:
    """Run multiple experiments and rank results."""

    def __init__(self):
        self.runner = ExperimentRunner()

    def run_batch(
        self,
        specs: List[ExperimentSpec],
        output_dir: Optional[Path] = None,
        rank_by: str = "sharpe",
    ) -> List[ExperimentResult]:
        """Run all specs, generate reports, rank by metric.

        Args:
            specs: List of experiment specifications.
            output_dir: Directory for reports. None = reports/.
            rank_by: "sharpe", "cagr", "oos_sharpe", or "tier".

        Returns:
            Results sorted by rank_by (best first).
        """
        if output_dir is None:
            output_dir = ROOT / "reports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: List[ExperimentResult] = []
        for i, spec in enumerate(specs, 1):
            logger.info("[%d/%d] Running %s ...", i, len(specs), spec.experiment_id)
            result = self.runner.run(spec)
            results.append(result)

            # Auto-generate per-experiment report
            report_name = f"{spec.experiment_id.lower().replace('-', '_')}_auto.html"
            self.runner.generate_report(result, output_dir / report_name)

            json_name = f"{spec.experiment_id.lower().replace('-', '_')}_auto.json"
            self.runner.save_json(result, output_dir / json_name)

            logger.info("  %s: Sharpe=%.2f, CAGR=%.1%%, Tier=%d",
                        spec.experiment_id, result.sharpe,
                        result.cagr * 100, result.tier)

        # Sort
        key_map = {
            "sharpe": lambda r: r.sharpe,
            "cagr": lambda r: r.cagr,
            "oos_sharpe": lambda r: r.oos_sharpe,
            "tier": lambda r: (-r.tier, r.sharpe),  # lower tier is better
        }
        sort_fn = key_map.get(rank_by, key_map["sharpe"])
        results.sort(key=sort_fn, reverse=True)

        # Generate batch summary report
        self._generate_batch_report(results, output_dir / "batch_summary.html")

        return results

    def _generate_batch_report(self, results: List[ExperimentResult],
                                output_path: Path) -> None:
        """Generate HTML summary comparing all experiments in the batch."""
        rows = ""
        for i, r in enumerate(results, 1):
            tc = {1: "#3fb950", 2: "#d29922", 3: "#f59e0b", 4: "#ef4444"}.get(r.tier, "#8b949e")
            rows += (
                f'<tr><td>{i}</td>'
                f'<td style="text-align:left"><strong>{r.spec.experiment_id}</strong></td>'
                f'<td style="text-align:left">{r.spec.name}</td>'
                f'<td>{r.spec.ticker}</td>'
                f'<td>{r.n_trades}</td>'
                f'<td style="color:{"#3fb950" if r.cagr > 0 else "#ef4444"}">{r.cagr:.1%}</td>'
                f'<td><strong>{r.sharpe:.2f}</strong></td>'
                f'<td>{r.max_dd:.1%}</td>'
                f'<td>{r.oos_sharpe:.2f}</td>'
                f'<td>{r.spy_corr:.3f}</td>'
                f'<td style="color:{tc}">Tier {r.tier}</td>'
                f'<td>{r.run_time_seconds:.1f}s</td></tr>\n'
            )

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Batch Experiment Summary</title>
<style>
body{{font-family:-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1{{color:#58a6ff;font-size:1.4rem}}
table{{width:100%;border-collapse:collapse;font-size:.82rem;margin:16px 0}}
th,td{{padding:5px 8px;text-align:right;border-bottom:1px solid #21262d}}
th{{background:#161b22;color:#8b949e;font-size:.72rem}}
td:first-child,th:first-child{{text-align:center}}
.note{{color:#8b949e;font-size:.82rem}}
</style></head><body>
<h1>Batch Experiment Summary</h1>
<p class="note">{len(results)} experiments &bull; Ranked by Sharpe &bull; {ts}</p>
<table>
<thead><tr><th>#</th><th>ID</th><th>Name</th><th>Ticker</th><th>Trades</th>
<th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>OOS SR</th><th>SPY Corr</th>
<th>Tier</th><th>Time</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""
        output_path.write_text(html, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Parameter sweep
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SweepResult:
    """One point in a parameter sweep."""
    params: Dict[str, Any]
    sharpe: float = 0.0
    cagr: float = 0.0
    max_dd: float = 0.0
    oos_sharpe: float = 0.0
    n_trades: int = 0
    tier: int = 4


class ParameterSweep:
    """Grid search over parameter space for a single strategy."""

    def __init__(self):
        self.runner = ExperimentRunner()

    def sweep(
        self,
        base_spec: ExperimentSpec,
        param_grid: Dict[str, List[Any]],
        rank_by: str = "oos_sharpe",
    ) -> List[SweepResult]:
        """Run grid search over param combinations.

        Args:
            base_spec: Template spec (experiment_id, name, etc.).
            param_grid: {"param_name": [val1, val2, ...], ...}.
            rank_by: Metric to sort results by.

        Returns:
            List of SweepResult sorted by rank_by (best first).
        """
        import itertools
        keys = sorted(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combos = list(itertools.product(*values))

        logger.info("Parameter sweep: %d combinations for %s",
                     len(combos), base_spec.experiment_id)

        results: List[SweepResult] = []
        for i, combo in enumerate(combos):
            param_dict = dict(zip(keys, combo))

            # Clone spec with new params
            merged_params = {**base_spec.params, **param_dict}
            spec = ExperimentSpec(
                experiment_id=f"{base_spec.experiment_id}_sweep_{i}",
                name=f"{base_spec.name} ({', '.join(f'{k}={v}' for k, v in param_dict.items())})",
                strategy_type=base_spec.strategy_type,
                ticker=base_spec.ticker,
                start_date=base_spec.start_date,
                end_date=base_spec.end_date,
                capital=base_spec.capital,
                params=merged_params,
                config_overrides=base_spec.config_overrides,
                validation=base_spec.validation,
                oos_start_year=base_spec.oos_start_year,
                custom_runner=base_spec.custom_runner,
                description=base_spec.description,
                data_source=base_spec.data_source,
                hypothesis=base_spec.hypothesis,
                success_criteria=base_spec.success_criteria,
            )

            try:
                result = self.runner.run(spec)
                results.append(SweepResult(
                    params=param_dict,
                    sharpe=result.sharpe,
                    cagr=result.cagr,
                    max_dd=result.max_dd,
                    oos_sharpe=result.oos_sharpe,
                    n_trades=result.n_trades,
                    tier=result.tier,
                ))
            except Exception as e:
                logger.warning("Sweep combo %d failed: %s", i, e)
                results.append(SweepResult(params=param_dict))

        # Sort
        key_map = {
            "sharpe": lambda r: r.sharpe,
            "cagr": lambda r: r.cagr,
            "oos_sharpe": lambda r: r.oos_sharpe,
        }
        results.sort(key=key_map.get(rank_by, key_map["oos_sharpe"]), reverse=True)

        return results


# ═══════════════════════════════════════════════════════════════════════════
# JSON registry integration
# ═══════════════════════════════════════════════════════════════════════════

REGISTRY_JSON = ROOT / "experiments" / "registry.json"


def update_json_registry(result: ExperimentResult,
                          path: Optional[Path] = None) -> None:
    """Add or update experiment in experiments/registry.json."""
    path = path or REGISTRY_JSON
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {"schema_version": "3.0", "last_updated": "", "experiments": {}}
    else:
        data = {"schema_version": "3.0", "last_updated": "", "experiments": {}}

    exp_id = result.spec.experiment_id
    data["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
    data["experiments"][exp_id] = {
        "id": exp_id,
        "name": result.spec.name,
        "strategy_type": result.spec.strategy_type,
        "ticker": result.spec.ticker,
        "status": {1: "live_ready", 2: "promising", 3: "marginal", 4: "dead"}.get(result.tier, "unknown"),
        "data_source": result.data_source,
        "created_date": result.timestamp[:10] if result.timestamp else "",
        "metrics": {
            "n_trades": result.n_trades,
            "sharpe": result.sharpe,
            "oos_sharpe": result.oos_sharpe,
            "cagr": round(result.cagr, 4),
            "max_dd": round(result.max_dd, 4),
            "win_rate": round(result.win_rate, 4),
            "spy_corr": round(result.spy_corr, 4),
        },
        "tier": result.tier,
        "verdict": result.verdict,
        "capacity": result.estimated_capacity,
        "hypothesis": result.spec.hypothesis,
        "description": result.spec.description,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Updated registry.json with %s", exp_id)


# ═══════════════════════════════════════════════════════════════════════════
# HTML report builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_html(r: ExperimentResult) -> str:
    spec = r.spec
    tier_colors = {1: "#3fb950", 2: "#d29922", 3: "#f59e0b", 4: "#ef4444"}
    vc = tier_colors.get(r.tier, "#8b949e")
    tier_labels = {1: "PRODUCTION READY", 2: "PROMISING", 3: "MARGINAL", 4: "DEAD"}

    # North Star checks
    ns_rows = ""
    for c in r.north_star_checks:
        icon = "&#10003;" if c.passed else "&#10007;"
        color = "#3fb950" if c.passed else "#ef4444"
        ns_rows += (
            f'<tr><td>{c.name}</td><td>{c.target}</td>'
            f'<td style="color:{color}"><strong>{c.actual}</strong></td>'
            f'<td style="color:{color}">{icon}</td></tr>\n'
        )

    # Yearly
    yr_rows = ""
    for yr in sorted(r.yearly.keys()):
        y = r.yearly[yr]
        is_oos = "OOS" if yr >= spec.oos_start_year else "IS"
        c = "#3fb950" if y["pnl"] > 0 else "#ef4444"
        yr_rows += (
            f'<tr><td>{yr} <span style="color:#8b949e;font-size:.7em">({is_oos})</span></td>'
            f'<td>{y["n"]}</td>'
            f'<td style="color:{c}">${y["pnl"]:,.0f}</td>'
            f'<td>{y["wr"]:.0%}</td>'
            f'<td>{y["sharpe"]:.2f}</td></tr>\n'
        )

    # Walk-forward windows
    wf_rows = ""
    for w in r.wf_windows:
        oos_c = "#3fb950" if w.oos_sharpe > 0 else "#ef4444"
        wf_rows += (
            f'<tr><td>{w.is_period}</td><td>{w.oos_period}</td>'
            f'<td>{w.is_trades}</td><td>{w.oos_trades}</td>'
            f'<td>{w.is_sharpe:.2f}</td>'
            f'<td style="color:{oos_c}"><strong>{w.oos_sharpe:.2f}</strong></td>'
            f'<td style="color:{"#3fb950" if w.oos_pnl > 0 else "#ef4444"}">${w.oos_pnl:,.0f}</td>'
            f'<td>{w.oos_wr:.0%}</td></tr>\n'
        )

    errors_html = ""
    if r.errors:
        errors_html = '<div class="finding fail"><h4>Errors</h4><ul>'
        for e in r.errors:
            errors_html += f"<li>{e}</li>"
        errors_html += "</ul></div>"

    now = r.timestamp or datetime.utcnow().isoformat()

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{spec.experiment_id}: {spec.name}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2,h3{{color:#58a6ff}}
.hero{{background:#161b22;border:2px solid {vc};border-radius:12px;padding:24px;text-align:center;margin:20px 0}}
.hero .big{{font-size:1.5em;font-weight:800;color:{vc}}}
.hero .sub{{color:#8b949e;margin-top:8px;font-size:.88em}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:#8b949e;font-size:.68em;text-transform:uppercase}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1em;margin-top:3px}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.82em}}
th,td{{padding:5px 8px;text-align:right;border-bottom:1px solid #21262d}}
th{{color:#8b949e;background:#161b22;font-size:.72em;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
tr:hover td{{background:#161b2280}}
.section{{margin:32px 0}}
.note{{color:#8b949e;font-size:.82em;margin:6px 0}}
.finding{{background:#161b22;border-left:4px solid #58a6ff;padding:14px;margin:14px 0;border-radius:4px;font-size:.85em}}
.finding h4{{margin:0 0 6px;color:#58a6ff;font-size:.9em}}
.win{{border-left-color:#3fb950}}.warn{{border-left-color:#f59e0b}}.fail{{border-left-color:#ef4444}}
</style></head><body>

<h1>{spec.experiment_id}: {spec.name}</h1>
<p class="note">{spec.strategy_type} on {spec.ticker} &bull; {spec.start_date} to {spec.end_date} &bull;
Data: {r.data_source} &bull; {now}</p>
{f'<p class="note">{spec.description}</p>' if spec.description else ''}

<div class="hero">
  <div class="big">TIER {r.tier} — {tier_labels.get(r.tier, "UNKNOWN")}</div>
  <div class="sub">{r.verdict}<br>
    {r.n_trades} trades &bull; PnL ${r.total_pnl:,.0f} &bull;
    CAGR {r.cagr:.1%} &bull; Sharpe {r.sharpe:.2f} &bull;
    DD {r.max_dd:.1%} &bull; WR {r.win_rate:.0%} &bull;
    OOS Sharpe {r.oos_sharpe:.2f} &bull; SPY corr {r.spy_corr:.3f}
  </div>
</div>

<div class="cards">
  <div class="c"><div class="l">CAGR</div><div class="v" style="color:{"#3fb950" if r.cagr > 0.1 else "#ef4444"}">{r.cagr:.1%}</div></div>
  <div class="c"><div class="l">Sharpe</div><div class="v">{r.sharpe:.2f}</div></div>
  <div class="c"><div class="l">Max DD</div><div class="v" style="color:#f59e0b">{r.max_dd:.1%}</div></div>
  <div class="c"><div class="l">Win Rate</div><div class="v">{r.win_rate:.0%}</div></div>
  <div class="c"><div class="l">OOS Sharpe</div><div class="v">{r.oos_sharpe:.2f}</div></div>
  <div class="c"><div class="l">SPY Corr</div><div class="v">{r.spy_corr:.3f}</div></div>
  <div class="c"><div class="l">Capacity</div><div class="v">{r.estimated_capacity}</div></div>
  <div class="c"><div class="l">Trades</div><div class="v">{r.n_trades}</div></div>
  <div class="c"><div class="l">PnL</div><div class="v" style="color:{"#3fb950" if r.total_pnl > 0 else "#ef4444"}">${r.total_pnl:,.0f}</div></div>
  <div class="c"><div class="l">North Star</div><div class="v" style="color:{vc}">{r.north_star_passed}/{r.north_star_total}</div></div>
</div>

{errors_html}

<div class="section">
<h2>North Star Evaluation</h2>
<table>
<thead><tr><th>Metric</th><th>Target</th><th>Actual</th><th>Pass?</th></tr></thead>
<tbody>{ns_rows}</tbody></table>
</div>

<div class="section">
<h2>Year-by-Year Performance</h2>
<table>
<thead><tr><th>Year</th><th>Trades</th><th>PnL</th><th>Win Rate</th><th>Sharpe</th></tr></thead>
<tbody>{yr_rows}</tbody></table>
</div>

<div class="section">
<h2>Walk-Forward Validation</h2>
<p class="note">IS = In-Sample, OOS = Out-of-Sample (year >= {spec.oos_start_year})</p>
<div class="cards" style="grid-template-columns:repeat(3,1fr)">
  <div class="c"><div class="l">IS Sharpe</div><div class="v">{r.is_sharpe:.2f}</div></div>
  <div class="c"><div class="l">OOS Sharpe</div><div class="v" style="color:{"#3fb950" if r.oos_sharpe > 0 else "#ef4444"}">{r.oos_sharpe:.2f}</div></div>
  <div class="c"><div class="l">WF Ratio</div><div class="v">{r.wf_ratio:.2f}</div></div>
</div>
<table>
<thead><tr><th>IS Period</th><th>OOS Period</th><th>IS Trades</th><th>OOS Trades</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS PnL</th><th>OOS WR</th></tr></thead>
<tbody>{wf_rows}</tbody></table>
</div>

<div class="note" style="margin-top:40px;text-align:center;border-top:1px solid #21262d;padding-top:16px">
  {spec.experiment_id} &bull; Generated by ExperimentRunner &bull; {now} &bull; PilotAI Compass
</div>
</body></html>"""
