"""
compass/experiment_launcher.py — Experiment launcher with config validation,
pre-flight checks, dry-run trade simulation, and rollback.

Provides:
  1. ExperimentLauncher class for starting/stopping experiments safely
  2. Config validation — YAML experiment config against required schema
  3. Pre-flight checks — data availability, model existence, feature
     compatibility, risk gate config
  4. Dry-run mode — simulates 10 trades without placing orders
  5. Rollback capability — reverts to previous config on failure
  6. HTML launch report with pre-flight checklist results

Usage:
    from compass.experiment_launcher import ExperimentLauncher

    launcher = ExperimentLauncher(config, experiment_id="EXP-400")
    report = launcher.preflight()
    if report.all_passed:
        launcher.start()
    # on failure:
    launcher.rollback()
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── Schema definition ─────────────────────────────────────────────────────────

STRATEGY_SCHEMA: Dict[str, Dict[str, Any]] = {
    "min_dte": {"type": int, "min": 0, "max": 365},
    "max_dte": {"type": int, "min": 1, "max": 365},
    "min_delta": {"type": (int, float), "min": 0.0, "max": 1.0},
    "max_delta": {"type": (int, float), "min": 0.0, "max": 1.0},
    "spread_width": {"type": (int, float), "min": 1, "max": 100},
}

RISK_SCHEMA: Dict[str, Dict[str, Any]] = {
    "max_risk_per_trade": {"type": (int, float), "min": 0.1, "max": 100.0},
    "max_positions": {"type": int, "min": 1, "max": 100},
    "profit_target": {"type": (int, float), "min": 1, "max": 100},
    "stop_loss_pct_of_width": {"type": (int, float), "min": 1, "max": 200},
}

BACKTEST_SCHEMA: Dict[str, Dict[str, Any]] = {
    "starting_capital": {"type": (int, float), "min": 1000, "max": 1e9},
    "commission_per_contract": {"type": (int, float), "min": 0.0, "max": 50.0},
}

TOP_LEVEL_REQUIRED = ("strategy", "risk", "backtest")

# Features that must be available in the feature pipeline for ML models.
REQUIRED_FEATURES = frozenset({
    "iv_rank", "vix_level", "rsi_14", "spy_return_5d",
    "dte", "delta", "spread_width", "net_credit",
})


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ValidationError:
    """A single config validation failure."""
    section: str
    field: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    name: str
    passed: bool
    message: str
    details: str = ""


@dataclass
class SimulatedTrade:
    """One simulated trade from the dry-run."""
    trade_id: int
    direction: str  # "bull_put" or "bear_call"
    contracts: int
    credit: float
    max_loss: float
    risk_pct: float
    pnl: float
    outcome: str  # "win" or "loss"
    risk_gate_passed: bool


@dataclass
class DryRunStep:
    """One step in a dry-run simulation."""
    stage: str
    status: str  # "ok", "warn", "fail", "skip"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DryRunResult:
    """Complete dry-run output."""
    steps: List[DryRunStep]
    simulated_trades: List[SimulatedTrade]
    would_trade: bool
    estimated_contracts: int
    estimated_risk_dollars: float
    risk_gate_result: str  # "pass", "block", "warn"
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Complete pre-flight report."""
    experiment_id: str
    timestamp: str
    validation_errors: List[ValidationError]
    checks: List[CheckResult]
    dry_run: Optional[DryRunResult]
    config_summary: Dict[str, Any]
    estimated_performance: Dict[str, Any]

    @property
    def all_passed(self) -> bool:
        errors = [e for e in self.validation_errors if e.severity == "error"]
        failed_checks = [c for c in self.checks if not c.passed]
        return len(errors) == 0 and len(failed_checks) == 0

    @property
    def n_errors(self) -> int:
        return sum(1 for e in self.validation_errors if e.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for e in self.validation_errors if e.severity == "warning")

    @property
    def n_checks_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_checks_total(self) -> int:
        return len(self.checks)


# ── Config validator ──────────────────────────────────────────────────────────

def validate_config(config: Dict[str, Any]) -> List[ValidationError]:
    """Validate experiment config against the required schema.

    Checks:
      - Top-level required sections present
      - All required fields exist within each section
      - Field types are correct
      - Numeric values are within valid ranges
      - Cross-field consistency (e.g. min_dte < max_dte)

    Args:
        config: Experiment config dict (typically loaded from YAML).

    Returns:
        List of ValidationError (empty = valid config).
    """
    errors: List[ValidationError] = []

    # Top-level sections
    for section in TOP_LEVEL_REQUIRED:
        if section not in config:
            errors.append(ValidationError(
                section="root", field=section,
                message=f"Missing required section '{section}'",
            ))

    # Strategy section
    strategy = config.get("strategy", {})
    if isinstance(strategy, dict):
        section_errors = _validate_section("strategy", strategy, STRATEGY_SCHEMA)
        errors.extend(section_errors)
        errored_fields = {e.field for e in section_errors}
        if (
            "min_dte" in strategy and "max_dte" in strategy
            and "min_dte" not in errored_fields and "max_dte" not in errored_fields
        ):
            if strategy["min_dte"] >= strategy["max_dte"]:
                errors.append(ValidationError(
                    section="strategy", field="min_dte/max_dte",
                    message=f"min_dte ({strategy['min_dte']}) must be < max_dte ({strategy['max_dte']})",
                ))
        if (
            "min_delta" in strategy and "max_delta" in strategy
            and "min_delta" not in errored_fields and "max_delta" not in errored_fields
        ):
            if strategy["min_delta"] >= strategy["max_delta"]:
                errors.append(ValidationError(
                    section="strategy", field="min_delta/max_delta",
                    message=f"min_delta ({strategy['min_delta']}) must be < max_delta ({strategy['max_delta']})",
                ))

    # Risk section
    risk = config.get("risk", {})
    if isinstance(risk, dict):
        errors.extend(_validate_section("risk", risk, RISK_SCHEMA))

    # Backtest section
    backtest = config.get("backtest", {})
    if isinstance(backtest, dict):
        errors.extend(_validate_section("backtest", backtest, BACKTEST_SCHEMA))

    return errors


def _validate_section(
    section_name: str,
    section: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]],
) -> List[ValidationError]:
    """Validate one config section against its schema."""
    errors: List[ValidationError] = []

    for field_name, constraints in schema.items():
        if field_name not in section:
            errors.append(ValidationError(
                section=section_name, field=field_name,
                message=f"Missing required field '{field_name}'",
            ))
            continue

        value = section[field_name]
        expected_type = constraints["type"]

        if not isinstance(value, expected_type):
            errors.append(ValidationError(
                section=section_name, field=field_name,
                message=f"Expected type {expected_type}, got {type(value).__name__}",
            ))
            continue

        min_val = constraints.get("min")
        max_val = constraints.get("max")
        if min_val is not None and value < min_val:
            errors.append(ValidationError(
                section=section_name, field=field_name,
                message=f"Value {value} below minimum {min_val}",
            ))
        if max_val is not None and value > max_val:
            errors.append(ValidationError(
                section=section_name, field=field_name,
                message=f"Value {value} above maximum {max_val}",
            ))

    return errors


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _check_training_data(config: Dict[str, Any], base_dir: Path) -> CheckResult:
    """Check that training data CSV exists."""
    data_dir = base_dir / "compass"
    candidates = list(data_dir.glob("training_data*.csv"))
    if candidates:
        return CheckResult(
            name="training_data",
            passed=True,
            message=f"Found {len(candidates)} training data file(s)",
            details=", ".join(p.name for p in candidates[:5]),
        )
    return CheckResult(
        name="training_data",
        passed=False,
        message="No training data CSV found in compass/",
    )


def _check_model_files(config: Dict[str, Any], base_dir: Path) -> CheckResult:
    """Check that model files exist (*.joblib or *.pkl)."""
    model_dirs = [base_dir / "compass", base_dir / "models", base_dir]
    model_files: List[Path] = []
    for d in model_dirs:
        if d.is_dir():
            model_files.extend(d.glob("*.joblib"))
            model_files.extend(d.glob("*.pkl"))

    if model_files:
        return CheckResult(
            name="model_files",
            passed=True,
            message=f"Found {len(model_files)} model file(s)",
            details=", ".join(p.name for p in model_files[:5]),
        )
    return CheckResult(
        name="model_files",
        passed=False,
        message="No model files found (*.joblib, *.pkl)",
    )


def _check_feature_pipeline(config: Dict[str, Any], base_dir: Path) -> CheckResult:
    """Check that feature pipeline module is importable."""
    feature_pipeline = base_dir / "compass" / "feature_pipeline.py"
    features = base_dir / "compass" / "features.py"
    if feature_pipeline.exists() or features.exists():
        return CheckResult(
            name="feature_pipeline",
            passed=True,
            message="Feature pipeline module found",
        )
    return CheckResult(
        name="feature_pipeline",
        passed=False,
        message="Feature pipeline not found",
    )


def _check_feature_compatibility(config: Dict[str, Any], base_dir: Path) -> CheckResult:
    """Check that training data contains required features for the ML model.

    Reads the header of the first training data CSV found and verifies that
    all REQUIRED_FEATURES are present as columns.
    """
    data_dir = base_dir / "compass"
    candidates = sorted(data_dir.glob("training_data*.csv"))
    if not candidates:
        return CheckResult(
            name="feature_compatibility",
            passed=False,
            message="Cannot check features — no training data CSV found",
        )

    try:
        with open(candidates[0]) as f:
            header_line = f.readline().strip()
        columns = frozenset(c.strip() for c in header_line.split(","))
    except Exception as e:
        return CheckResult(
            name="feature_compatibility",
            passed=False,
            message=f"Failed to read training data header: {e}",
        )

    missing = REQUIRED_FEATURES - columns
    if missing:
        return CheckResult(
            name="feature_compatibility",
            passed=False,
            message=f"Missing {len(missing)} required feature(s): {', '.join(sorted(missing))}",
        )

    return CheckResult(
        name="feature_compatibility",
        passed=True,
        message=f"All {len(REQUIRED_FEATURES)} required features present",
        details=", ".join(sorted(REQUIRED_FEATURES)),
    )


def _check_risk_gates(config: Dict[str, Any], base_dir: Path) -> CheckResult:
    """Check that risk gate configuration is valid."""
    risk = config.get("risk", {})
    if not risk:
        return CheckResult(
            name="risk_gates",
            passed=False,
            message="No risk configuration found",
        )

    critical = ["max_risk_per_trade", "max_positions"]
    missing = [k for k in critical if k not in risk]
    if missing:
        return CheckResult(
            name="risk_gates",
            passed=False,
            message=f"Missing critical risk params: {', '.join(missing)}",
        )

    return CheckResult(
        name="risk_gates",
        passed=True,
        message="Risk gates configured",
        details=f"max_risk={risk.get('max_risk_per_trade')}%, max_pos={risk.get('max_positions')}",
    )


def _check_hedge_params(config: Dict[str, Any], base_dir: Path) -> CheckResult:
    """Check that hedge parameters are set (optional but recommended)."""
    risk = config.get("risk", {})
    has_stop = "stop_loss_pct_of_width" in risk
    has_target = "profit_target" in risk

    if has_stop and has_target:
        return CheckResult(
            name="hedge_params",
            passed=True,
            message="Hedge params configured",
            details=f"target={risk['profit_target']}%, stop={risk['stop_loss_pct_of_width']}%",
        )
    missing = []
    if not has_stop:
        missing.append("stop_loss_pct_of_width")
    if not has_target:
        missing.append("profit_target")
    return CheckResult(
        name="hedge_params",
        passed=False,
        message=f"Missing hedge params: {', '.join(missing)}",
    )


DEFAULT_CHECKS: List[Callable] = [
    _check_training_data,
    _check_model_files,
    _check_feature_pipeline,
    _check_feature_compatibility,
    _check_risk_gates,
    _check_hedge_params,
]


# ── Trade simulation ──────────────────────────────────────────────────────────

def _simulate_trades(
    config: Dict[str, Any],
    n_trades: int = 10,
    seed: int = 42,
) -> List[SimulatedTrade]:
    """Simulate n_trades to test the pipeline end-to-end without orders.

    Uses config parameters to generate realistic synthetic trades:
      - Contracts sized from risk budget
      - Credits drawn from spread_width distribution
      - Outcomes based on ~65% base win rate (typical credit spread)
      - Risk gate applied per trade
    """
    rng = np.random.RandomState(seed)
    strategy = config.get("strategy", {})
    risk = config.get("risk", {})
    backtest = config.get("backtest", {})

    capital = backtest.get("starting_capital", 100_000)
    max_risk_pct = risk.get("max_risk_per_trade", 5.0) / 100.0
    spread_width = strategy.get("spread_width", 5.0)
    max_positions = risk.get("max_positions", 6)
    masterplan_risk_limit = 0.05  # 5% MASTERPLAN hard cap

    trades: List[SimulatedTrade] = []
    for i in range(n_trades):
        direction = "bull_put" if rng.random() > 0.4 else "bear_call"

        # Credit: typically 20-35% of spread width
        credit_frac = rng.uniform(0.20, 0.35)
        credit = round(spread_width * credit_frac, 2)
        max_loss = round((spread_width - credit) * 100, 2)

        # Contracts from risk budget
        risk_budget = capital * min(max_risk_pct, masterplan_risk_limit)
        contracts = max(1, int(risk_budget / max(max_loss, 1)))

        risk_pct = (max_loss * contracts) / capital if capital > 0 else 0.0

        # Risk gate: block if exceeds per-trade or position limit
        risk_gate_passed = (
            risk_pct <= masterplan_risk_limit
            and i < max_positions  # simplistic position count check
        )

        # Outcome: ~65% win rate with random P&L
        is_win = rng.random() < 0.65
        if is_win:
            pnl = round(credit * 100 * contracts, 2)
        else:
            loss_frac = rng.uniform(0.3, 1.0)
            pnl = round(-max_loss * contracts * loss_frac, 2)

        trades.append(SimulatedTrade(
            trade_id=i + 1,
            direction=direction,
            contracts=contracts,
            credit=credit,
            max_loss=max_loss,
            risk_pct=round(risk_pct * 100, 2),
            pnl=pnl,
            outcome="win" if is_win else "loss",
            risk_gate_passed=risk_gate_passed,
        ))

    return trades


# ── Dry-run simulation ────────────────────────────────────────────────────────

def simulate_dry_run(
    config: Dict[str, Any],
    base_dir: Path,
    n_trades: int = 10,
) -> DryRunResult:
    """Simulate the full pipeline without placing trades.

    Stages:
      1. Load config
      2. Build features (check pipeline availability)
      3. Run model (check model files)
      4. Check sizing (compute hypothetical position size)
      5. Apply risk gates (evaluate against limits)
      6. Simulate trades (generate n_trades synthetic trades)
      7. Summary

    Args:
        config: Experiment config dict.
        base_dir: Project root directory.
        n_trades: Number of trades to simulate (default 10).

    Returns:
        DryRunResult with pipeline steps and simulated trades.
    """
    steps: List[DryRunStep] = []

    # 1. Load config
    strategy = config.get("strategy", {})
    risk = config.get("risk", {})
    backtest = config.get("backtest", {})
    steps.append(DryRunStep(
        stage="load_config",
        status="ok",
        message="Config loaded successfully",
        details={"sections": list(config.keys())},
    ))

    # 2. Build features
    feature_file = base_dir / "compass" / "feature_pipeline.py"
    features_file = base_dir / "compass" / "features.py"
    if feature_file.exists() or features_file.exists():
        steps.append(DryRunStep(
            stage="build_features",
            status="ok",
            message="Feature pipeline available",
        ))
    else:
        steps.append(DryRunStep(
            stage="build_features",
            status="fail",
            message="Feature pipeline not found — cannot generate features",
        ))

    # 3. Run model
    model_found = False
    for d in [base_dir / "compass", base_dir / "models", base_dir]:
        if d.is_dir():
            if list(d.glob("*.joblib")) or list(d.glob("*.pkl")):
                model_found = True
                break
    if model_found:
        steps.append(DryRunStep(
            stage="run_model",
            status="ok",
            message="ML model available for inference",
        ))
    else:
        steps.append(DryRunStep(
            stage="run_model",
            status="warn",
            message="No ML model found — would fall back to rule-based signals",
        ))

    # 4. Check sizing
    capital = backtest.get("starting_capital", 100_000)
    max_risk_pct = risk.get("max_risk_per_trade", 5.0)
    spread_width = strategy.get("spread_width", 5)
    risk_budget = capital * (max_risk_pct / 100.0)
    max_loss_per_contract = spread_width * 100 * 0.7
    estimated_contracts = max(1, int(risk_budget / max(max_loss_per_contract, 1)))

    steps.append(DryRunStep(
        stage="check_sizing",
        status="ok",
        message=f"Estimated {estimated_contracts} contracts at ${risk_budget:,.0f} risk budget",
        details={
            "capital": capital,
            "risk_pct": max_risk_pct,
            "risk_budget": risk_budget,
            "spread_width": spread_width,
            "estimated_contracts": estimated_contracts,
        },
    ))

    # 5. Apply risk gates
    max_positions = risk.get("max_positions", 6)
    max_portfolio_risk_pct = risk.get("max_portfolio_risk_pct", 40)
    single_trade_exposure = max_risk_pct / 100.0

    risk_gate_status = "pass"
    risk_gate_messages = []

    if single_trade_exposure > 0.05:
        risk_gate_messages.append(
            f"Per-trade risk {max_risk_pct}% exceeds MASTERPLAN 5% limit"
        )
        risk_gate_status = "block"

    if max_portfolio_risk_pct > 40:
        risk_gate_messages.append(
            f"Portfolio heat {max_portfolio_risk_pct}% exceeds 40% cap"
        )
        risk_gate_status = "block"

    if not risk_gate_messages:
        risk_gate_messages.append("All risk gates pass")

    steps.append(DryRunStep(
        stage="risk_gates",
        status="ok" if risk_gate_status == "pass" else "fail",
        message="; ".join(risk_gate_messages),
        details={
            "risk_gate_result": risk_gate_status,
            "per_trade_pct": max_risk_pct,
            "max_positions": max_positions,
            "portfolio_heat_cap": max_portfolio_risk_pct,
        },
    ))

    # 6. Simulate trades
    simulated = _simulate_trades(config, n_trades=n_trades)
    n_wins = sum(1 for t in simulated if t.outcome == "win")
    n_blocked = sum(1 for t in simulated if not t.risk_gate_passed)
    total_pnl = sum(t.pnl for t in simulated)
    steps.append(DryRunStep(
        stage="simulate_trades",
        status="ok",
        message=f"Simulated {len(simulated)} trades: {n_wins}W/{len(simulated)-n_wins}L, "
                f"P&L=${total_pnl:,.0f}, {n_blocked} blocked by risk gate",
        details={
            "n_trades": len(simulated),
            "n_wins": n_wins,
            "n_blocked": n_blocked,
            "total_pnl": total_pnl,
        },
    ))

    # 7. Summary
    would_trade = risk_gate_status == "pass" and all(
        s.status in ("ok", "warn") for s in steps
    )
    steps.append(DryRunStep(
        stage="summary",
        status="ok" if would_trade else "fail",
        message="Pipeline would proceed to trade" if would_trade else "Pipeline blocked — see failures above",
    ))

    return DryRunResult(
        steps=steps,
        simulated_trades=simulated,
        would_trade=would_trade,
        estimated_contracts=estimated_contracts,
        estimated_risk_dollars=risk_budget,
        risk_gate_result=risk_gate_status,
        summary={
            "n_steps": len(steps),
            "n_ok": sum(1 for s in steps if s.status == "ok"),
            "n_warn": sum(1 for s in steps if s.status == "warn"),
            "n_fail": sum(1 for s in steps if s.status == "fail"),
            "simulated_pnl": total_pnl,
            "simulated_win_rate": n_wins / max(len(simulated), 1),
        },
    )


# ── Launcher class ────────────────────────────────────────────────────────────

class ExperimentLauncher:
    """Validates config, runs pre-flight checks, manages experiment lifecycle.

    Supports starting/stopping experiments safely, dry-run with simulated
    trades, and automatic rollback to a previous config on failure.

    Args:
        config: Experiment config dict (typically from YAML).
        experiment_id: Experiment identifier.
        base_dir: Project root directory (auto-detected if None).
        checks: Optional custom list of check functions.
        backtest_metrics: Optional dict of last backtest results for
            estimated performance display.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        experiment_id: str = "EXP-NEW",
        base_dir: Optional[Path] = None,
        checks: Optional[List] = None,
        backtest_metrics: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.experiment_id = experiment_id
        self.base_dir = base_dir or ROOT
        self.checks = checks if checks is not None else DEFAULT_CHECKS
        self.backtest_metrics = backtest_metrics or {}

        # Lifecycle state
        self._state: str = "idle"  # idle -> preflight -> running -> stopped
        self._previous_config: Optional[Dict[str, Any]] = None
        self._start_time: Optional[str] = None
        self._stop_time: Optional[str] = None
        self._stop_reason: Optional[str] = None
        self._last_report: Optional[PreflightReport] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current lifecycle state: idle, preflight, running, stopped."""
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == "running"

    @property
    def previous_config(self) -> Optional[Dict[str, Any]]:
        return self._previous_config

    # ─────────────────────────────────────────────────────────────────────────
    # Validation & checks
    # ─────────────────────────────────────────────────────────────────────────

    def validate(self) -> List[ValidationError]:
        """Validate config against schema."""
        return validate_config(self.config)

    def run_checks(self) -> List[CheckResult]:
        """Run all pre-flight checks."""
        results = []
        for check_fn in self.checks:
            try:
                result = check_fn(self.config, self.base_dir)
                results.append(result)
            except Exception as e:
                logger.warning("Check %s failed: %s", check_fn.__name__, e)
                results.append(CheckResult(
                    name=check_fn.__name__.lstrip("_check_"),
                    passed=False,
                    message=f"Check raised exception: {e}",
                ))
        return results

    def dry_run(self, n_trades: int = 10) -> DryRunResult:
        """Run dry-run simulation with n_trades simulated trades."""
        return simulate_dry_run(self.config, self.base_dir, n_trades=n_trades)

    def preflight(self, include_dry_run: bool = True) -> PreflightReport:
        """Run complete pre-flight: validate + checks + optional dry run.

        Args:
            include_dry_run: Whether to include dry-run simulation.

        Returns:
            PreflightReport with all results.
        """
        self._state = "preflight"
        validation_errors = self.validate()
        checks = self.run_checks()
        dry_run_result = self.dry_run() if include_dry_run else None

        strategy = self.config.get("strategy", {})
        risk = self.config.get("risk", {})
        backtest = self.config.get("backtest", {})

        config_summary = {
            "experiment_id": self.experiment_id,
            "strategy": {
                "dte_range": f"{strategy.get('min_dte', '?')}-{strategy.get('max_dte', '?')}",
                "delta_range": f"{strategy.get('min_delta', '?')}-{strategy.get('max_delta', '?')}",
                "spread_width": strategy.get("spread_width", "?"),
            },
            "risk": {
                "max_risk_per_trade": f"{risk.get('max_risk_per_trade', '?')}%",
                "max_positions": risk.get("max_positions", "?"),
                "profit_target": f"{risk.get('profit_target', '?')}%",
                "stop_loss": f"{risk.get('stop_loss_pct_of_width', '?')}%",
            },
            "backtest": {
                "starting_capital": f"${backtest.get('starting_capital', 100000):,.0f}",
                "commission": f"${backtest.get('commission_per_contract', 0.65):.2f}",
            },
        }

        report = PreflightReport(
            experiment_id=self.experiment_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            validation_errors=validation_errors,
            checks=checks,
            dry_run=dry_run_result,
            config_summary=config_summary,
            estimated_performance=self.backtest_metrics,
        )
        self._last_report = report
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle: start / stop / rollback
    # ─────────────────────────────────────────────────────────────────────────

    def start(self, force: bool = False) -> PreflightReport:
        """Start the experiment after running pre-flight checks.

        Saves current config as the rollback snapshot before transitioning
        to the running state.

        Args:
            force: If True, start even if pre-flight has warnings (but not
                   errors). Default False — all checks must pass.

        Returns:
            PreflightReport from the pre-flight run.

        Raises:
            RuntimeError: If already running or pre-flight fails.
        """
        if self._state == "running":
            raise RuntimeError(
                f"Experiment {self.experiment_id} is already running"
            )

        report = self.preflight()

        has_errors = report.n_errors > 0
        has_failed_checks = report.n_checks_passed < report.n_checks_total

        if has_errors:
            self._state = "idle"
            raise RuntimeError(
                f"Cannot start: {report.n_errors} validation error(s)"
            )

        if has_failed_checks and not force:
            self._state = "idle"
            raise RuntimeError(
                f"Cannot start: {report.n_checks_total - report.n_checks_passed} "
                f"pre-flight check(s) failed. Use force=True to override."
            )

        # Save rollback snapshot
        self._previous_config = copy.deepcopy(self.config)
        self._start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._state = "running"

        logger.info(
            "Experiment %s started at %s", self.experiment_id, self._start_time
        )
        return report

    def stop(self, reason: str = "manual") -> None:
        """Stop the experiment gracefully.

        Args:
            reason: Why the experiment was stopped (logged and stored).

        Raises:
            RuntimeError: If experiment is not running.
        """
        if self._state != "running":
            raise RuntimeError(
                f"Experiment {self.experiment_id} is not running (state={self._state})"
            )

        self._stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stop_reason = reason
        self._state = "stopped"

        logger.info(
            "Experiment %s stopped at %s: %s",
            self.experiment_id, self._stop_time, reason,
        )

    def rollback(self) -> Dict[str, Any]:
        """Revert to the previous config snapshot.

        Restores the config that was active before ``start()`` was called.
        Also transitions the experiment to stopped state if it was running.

        Returns:
            The restored config dict.

        Raises:
            RuntimeError: If no previous config is available.
        """
        if self._previous_config is None:
            raise RuntimeError("No previous config to rollback to")

        old = self.config
        self.config = copy.deepcopy(self._previous_config)

        if self._state == "running":
            self._stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._stop_reason = "rollback"
            self._state = "stopped"

        logger.info(
            "Experiment %s rolled back to previous config", self.experiment_id
        )
        return self.config

    def update_config(self, new_config: Dict[str, Any]) -> List[ValidationError]:
        """Update the experiment config after validation.

        Saves the current config as the rollback snapshot, validates the
        new config, and applies it only if validation passes.

        Args:
            new_config: New config dict to apply.

        Returns:
            List of validation errors (empty = config applied).

        Raises:
            RuntimeError: If experiment is currently running.
        """
        if self._state == "running":
            raise RuntimeError(
                "Cannot update config while experiment is running. Stop first."
            )

        errors = validate_config(new_config)
        hard_errors = [e for e in errors if e.severity == "error"]
        if hard_errors:
            return errors

        self._previous_config = copy.deepcopy(self.config)
        self.config = new_config
        logger.info("Config updated for %s", self.experiment_id)
        return errors

    # ─────────────────────────────────────────────────────────────────────────
    # HTML report
    # ─────────────────────────────────────────────────────────────────────────

    def generate_html(self, report: PreflightReport) -> str:
        """Generate HTML launch report.

        Sections:
          - Pre-flight status (pass/fail badge)
          - Validation errors table
          - Pre-flight checks checklist
          - Dry-run pipeline trace with simulated trades
          - Config summary
          - Risk parameters
          - Estimated performance (from last backtest)
        """
        now = report.timestamp
        status_badge = (
            '<span style="background:#16a34a;color:#fff;padding:4px 12px;'
            'border-radius:4px;font-weight:700">READY</span>'
            if report.all_passed else
            '<span style="background:#dc2626;color:#fff;padding:4px 12px;'
            'border-radius:4px;font-weight:700">NOT READY</span>'
        )

        # ── Validation errors ────────────────────────────────────────────
        val_rows = ""
        for ve in report.validation_errors:
            sev_cls = "bad" if ve.severity == "error" else "warn"
            val_rows += (
                f"<tr><td class='{sev_cls}'>{ve.severity.upper()}</td>"
                f"<td>{ve.section}</td><td>{ve.field}</td>"
                f"<td>{ve.message}</td></tr>\n"
            )
        if val_rows:
            val_section = f"""
<div class="section">
<h2>Validation Errors ({report.n_errors} errors, {report.n_warnings} warnings)</h2>
<table>
<thead><tr><th>Severity</th><th>Section</th><th>Field</th><th>Message</th></tr></thead>
<tbody>{val_rows}</tbody>
</table>
</div>"""
        else:
            val_section = (
                '<div class="section"><h2>Validation</h2>'
                '<p class="good">All config fields valid.</p></div>'
            )

        # ── Pre-flight checks ────────────────────────────────────────────
        check_rows = ""
        for ck in report.checks:
            status = "PASS" if ck.passed else "FAIL"
            cls = "good" if ck.passed else "bad"
            details = f" — {ck.details}" if ck.details else ""
            check_rows += (
                f"<tr><td class='{cls}'>{status}</td>"
                f"<td>{ck.name}</td><td>{ck.message}{details}</td></tr>\n"
            )

        # ── Dry-run trace ────────────────────────────────────────────────
        dry_run_html = ""
        if report.dry_run:
            dr = report.dry_run
            dr_rows = ""
            for step in dr.steps:
                cls = {"ok": "good", "warn": "warn", "fail": "bad",
                       "skip": "neutral"}.get(step.status, "")
                dr_rows += (
                    f"<tr><td>{step.stage}</td>"
                    f"<td class='{cls}'>{step.status.upper()}</td>"
                    f"<td>{step.message}</td></tr>\n"
                )

            trade_badge = (
                '<span class="good">YES</span>' if dr.would_trade
                else '<span class="bad">NO</span>'
            )

            # Simulated trades table
            trade_rows = ""
            for t in dr.simulated_trades:
                pnl_cls = "good" if t.pnl > 0 else "bad"
                gate_cls = "good" if t.risk_gate_passed else "bad"
                trade_rows += (
                    f"<tr><td>{t.trade_id}</td><td>{t.direction}</td>"
                    f"<td>{t.contracts}</td><td>${t.credit:.2f}</td>"
                    f"<td>${t.max_loss:.2f}</td><td>{t.risk_pct:.1f}%</td>"
                    f"<td class='{pnl_cls}'>${t.pnl:,.2f}</td>"
                    f"<td>{t.outcome}</td>"
                    f"<td class='{gate_cls}'>{'PASS' if t.risk_gate_passed else 'BLOCK'}</td></tr>\n"
                )

            sim_pnl = dr.summary.get("simulated_pnl", 0)
            sim_wr = dr.summary.get("simulated_win_rate", 0)

            dry_run_html = f"""
<div class="section">
<h2>Dry Run</h2>
<div class="kpi-row">
  <div class="kpi"><div class="value">{trade_badge}</div><div class="label">Would Trade</div></div>
  <div class="kpi"><div class="value">{dr.estimated_contracts}</div><div class="label">Est. Contracts</div></div>
  <div class="kpi"><div class="value">${dr.estimated_risk_dollars:,.0f}</div><div class="label">Risk Budget</div></div>
  <div class="kpi"><div class="value {'good' if dr.risk_gate_result == 'pass' else 'bad'}">{dr.risk_gate_result.upper()}</div><div class="label">Risk Gate</div></div>
</div>
<table>
<thead><tr><th>Stage</th><th>Status</th><th>Message</th></tr></thead>
<tbody>{dr_rows}</tbody>
</table>
<h3>Simulated Trades ({len(dr.simulated_trades)} trades, WR={sim_wr:.0%}, P&amp;L=${sim_pnl:,.0f})</h3>
<table>
<thead><tr><th>#</th><th>Dir</th><th>Qty</th><th>Credit</th><th>Max Loss</th><th>Risk%</th><th>P&amp;L</th><th>Result</th><th>Gate</th></tr></thead>
<tbody>{trade_rows}</tbody>
</table>
</div>"""

        # ── Config summary ───────────────────────────────────────────────
        cs = report.config_summary
        config_rows = ""
        for section_name, section_data in cs.items():
            if isinstance(section_data, dict):
                for k, v in section_data.items():
                    config_rows += (
                        f"<tr><td>{section_name}.{k}</td><td>{v}</td></tr>\n"
                    )
            else:
                config_rows += (
                    f"<tr><td>{section_name}</td><td>{section_data}</td></tr>\n"
                )

        # ── Estimated performance ────────────────────────────────────────
        perf_html = ""
        if report.estimated_performance:
            perf_rows = ""
            for k, v in report.estimated_performance.items():
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    perf_rows += f"<tr><td>{label}</td><td>{v:.2f}</td></tr>\n"
                else:
                    perf_rows += f"<tr><td>{label}</td><td>{v}</td></tr>\n"
            perf_html = f"""
<div class="section">
<h2>Estimated Performance (Last Backtest)</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{perf_rows}</tbody>
</table>
</div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Experiment Launch Report — {report.experiment_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  h3 {{ color: #475569; margin-top: 1.5em; font-size: 1em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .good {{ color: #16a34a; }}
  .bad {{ color: #dc2626; }}
  .warn {{ color: #d97706; }}
  .neutral {{ color: #64748b; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .section {{ margin-bottom: 2.5em; }}
</style>
</head>
<body>
<h1>Experiment Launch Report — {report.experiment_id}</h1>
<div class="meta">Generated: {now} | Status: {status_badge}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value {'good' if report.all_passed else 'bad'}">{'READY' if report.all_passed else 'BLOCKED'}</div><div class="label">Launch Status</div></div>
  <div class="kpi"><div class="value {'good' if report.n_errors == 0 else 'bad'}">{report.n_errors}</div><div class="label">Validation Errors</div></div>
  <div class="kpi"><div class="value">{report.n_checks_passed}/{report.n_checks_total}</div><div class="label">Checks Passed</div></div>
  <div class="kpi"><div class="value">{report.n_warnings}</div><div class="label">Warnings</div></div>
</div>

{val_section}

<div class="section">
<h2>Pre-Flight Checks ({report.n_checks_passed}/{report.n_checks_total} passed)</h2>
<table>
<thead><tr><th>Status</th><th>Check</th><th>Details</th></tr></thead>
<tbody>{check_rows}</tbody>
</table>
</div>

{dry_run_html}

<div class="section">
<h2>Config Summary</h2>
<table>
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>{config_rows}</tbody>
</table>
</div>

{perf_html}

</body>
</html>"""
        return html

    def generate_report_file(
        self,
        output_path: str = "reports/launch_report.html",
        include_dry_run: bool = True,
    ) -> PreflightReport:
        """Run preflight and write HTML report to disk."""
        report = self.preflight(include_dry_run=include_dry_run)
        html = self.generate_html(report)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

        logger.info("Launch report written to %s", output_path)
        return report
