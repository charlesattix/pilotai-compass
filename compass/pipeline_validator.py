"""
Production pipeline validator — end-to-end validation of the COMPASS pipeline.

Stages validated:
  1. Data ingestion     — schema, types, freshness, nulls
  2. Feature engineering — required columns, ranges, NaN ratio
  3. Signal generation  — score bounds, distribution, coverage
  4. Model prediction   — staleness, output shape, calibration
  5. Position sizing    — contract bounds, notional limits
  6. Risk checks        — drawdown gate, exposure limit, regime filter
  7. Order generation   — valid sides, price sanity, duplicate detection
  8. Hedge overlay      — hedge ratio bounds, notional cap

Each stage produces a StageResult with pass/fail, timing, errors, and
remediation suggestions.  Generates an HTML pass/fail report.

Runnable as a pre-deploy check: all stages must PASS for green light.

Usage::

    from compass.pipeline_validator import PipelineValidator
    pv = PipelineValidator(trades_df)
    result = pv.validate()
    PipelineValidator.generate_report(result)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "pipeline_validation.html"

STAGE_NAMES = [
    "data_ingestion",
    "feature_engineering",
    "signal_generation",
    "model_prediction",
    "position_sizing",
    "risk_checks",
    "order_generation",
    "hedge_overlay",
]


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class StageError:
    """A single validation error."""

    code: str
    message: str
    severity: str          # "error", "warning", "info"
    remediation: str


@dataclass
class StageResult:
    """Validation result for one pipeline stage."""

    stage: str
    passed: bool
    elapsed_ms: float
    n_checks: int
    n_passed: int
    n_failed: int
    n_warnings: int
    errors: List[StageError]
    input_shape: Tuple[int, ...] = (0,)
    output_shape: Tuple[int, ...] = (0,)


@dataclass
class KillSwitchResult:
    """Kill-switch validation."""

    triggered: bool
    reason: str
    checks: List[Tuple[str, bool, str]]  # (check_name, passed, detail)


@dataclass
class ValidationResult:
    """Full pipeline validation result."""

    stages: List[StageResult]
    kill_switch: KillSwitchResult
    overall_pass: bool
    total_elapsed_ms: float
    n_stages: int
    n_stages_passed: int
    n_total_errors: int
    n_total_warnings: int
    timestamp: str


# ── Validation checks ────────────────────────────────────────────────────


def check_schema(
    df: pd.DataFrame,
    required_columns: List[str],
) -> List[StageError]:
    """Check that required columns exist."""
    errors: List[StageError] = []
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(StageError(
            code="SCHEMA_MISSING",
            message=f"Missing columns: {sorted(missing)}",
            severity="error",
            remediation="Add missing columns to data source or feature pipeline.",
        ))
    return errors


def check_types(
    df: pd.DataFrame,
    numeric_columns: List[str],
) -> List[StageError]:
    """Check that numeric columns are actually numeric."""
    errors: List[StageError] = []
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(StageError(
                code="TYPE_MISMATCH",
                message=f"Column '{col}' is not numeric (dtype={df[col].dtype})",
                severity="error",
                remediation=f"Cast '{col}' to float/int in feature pipeline.",
            ))
    return errors


def check_nulls(
    df: pd.DataFrame,
    max_null_pct: float = 0.10,
) -> List[StageError]:
    """Check for excessive nulls."""
    errors: List[StageError] = []
    for col in df.columns:
        null_pct = df[col].isna().mean()
        if null_pct > max_null_pct:
            sev = "error" if null_pct > 0.5 else "warning"
            errors.append(StageError(
                code="EXCESSIVE_NULLS",
                message=f"Column '{col}' has {null_pct:.1%} nulls",
                severity=sev,
                remediation=f"Impute or drop nulls in '{col}'.",
            ))
    return errors


def check_freshness(
    df: pd.DataFrame,
    date_col: str = "entry_date",
    max_stale_days: int = 30,
) -> List[StageError]:
    """Check data freshness."""
    errors: List[StageError] = []
    if date_col not in df.columns:
        return errors
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        latest = dates.max()
        if pd.isna(latest):
            errors.append(StageError(
                code="NO_VALID_DATES",
                message=f"No valid dates in '{date_col}'",
                severity="error",
                remediation="Fix date parsing in data ingestion.",
            ))
        else:
            age = (pd.Timestamp.now() - latest).days
            if age > max_stale_days:
                errors.append(StageError(
                    code="STALE_DATA",
                    message=f"Latest date is {age} days old (max {max_stale_days})",
                    severity="warning",
                    remediation="Update data source with recent observations.",
                ))
    except Exception as e:
        errors.append(StageError(
            code="DATE_PARSE_ERROR",
            message=str(e),
            severity="error",
            remediation="Fix date format in data source.",
        ))
    return errors


def check_value_range(
    series: pd.Series,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> List[StageError]:
    """Check values are within expected range."""
    errors: List[StageError] = []
    vals = series.dropna()
    if len(vals) == 0:
        return errors
    if min_val is not None and float(vals.min()) < min_val:
        errors.append(StageError(
            code="RANGE_VIOLATION",
            message=f"'{name}' has values below {min_val} (min={vals.min():.4f})",
            severity="error",
            remediation=f"Clip or filter '{name}' to [{min_val}, {max_val}].",
        ))
    if max_val is not None and float(vals.max()) > max_val:
        errors.append(StageError(
            code="RANGE_VIOLATION",
            message=f"'{name}' has values above {max_val} (max={vals.max():.4f})",
            severity="error",
            remediation=f"Clip or filter '{name}' to [{min_val}, {max_val}].",
        ))
    return errors


def check_duplicates(
    df: pd.DataFrame,
    key_columns: List[str],
) -> List[StageError]:
    """Check for duplicate rows on key columns."""
    errors: List[StageError] = []
    available = [c for c in key_columns if c in df.columns]
    if not available:
        return errors
    n_dup = df.duplicated(subset=available).sum()
    if n_dup > 0:
        errors.append(StageError(
            code="DUPLICATES",
            message=f"{n_dup} duplicate rows on {available}",
            severity="warning",
            remediation="Deduplicate input data on key columns.",
        ))
    return errors


def check_distribution(
    series: pd.Series,
    name: str,
    expected_mean_range: Optional[Tuple[float, float]] = None,
) -> List[StageError]:
    """Check distribution is within expected parameters."""
    errors: List[StageError] = []
    vals = series.dropna()
    if len(vals) < 5:
        return errors
    if expected_mean_range:
        mu = float(vals.mean())
        lo, hi = expected_mean_range
        if mu < lo or mu > hi:
            errors.append(StageError(
                code="DISTRIBUTION_SHIFT",
                message=f"'{name}' mean={mu:.4f} outside expected [{lo}, {hi}]",
                severity="warning",
                remediation=f"Investigate distribution shift in '{name}'.",
            ))
    return errors


# ── Kill switch ──────────────────────────────────────────────────────────


def evaluate_kill_switch(
    stages: List[StageResult],
    max_errors: int = 3,
) -> KillSwitchResult:
    """Determine if kill switch should be triggered."""
    checks: List[Tuple[str, bool, str]] = []
    total_errors = sum(s.n_failed for s in stages)
    critical_fail = any(not s.passed for s in stages[:3])  # first 3 stages critical

    checks.append(("total_errors_below_limit",
                    total_errors <= max_errors,
                    f"{total_errors} errors (max {max_errors})"))
    checks.append(("critical_stages_pass",
                    not critical_fail,
                    "data/feature/signal stages must all pass"))
    checks.append(("all_stages_ran",
                    len(stages) == len(STAGE_NAMES),
                    f"{len(stages)}/{len(STAGE_NAMES)} stages ran"))

    triggered = total_errors > max_errors or critical_fail
    reason = ""
    if triggered:
        failed = [c[0] for c in checks if not c[1]]
        reason = f"Kill switch: {', '.join(failed)}"

    return KillSwitchResult(triggered=triggered, reason=reason, checks=checks)


# ── Stage validators ─────────────────────────────────────────────────────


def validate_data_ingestion(df: pd.DataFrame) -> StageResult:
    """Stage 1: Validate raw data ingestion."""
    t0 = time.monotonic()
    errors: List[StageError] = []
    n_checks = 0

    # Schema
    required = ["entry_date", "exit_date", "pnl"]
    errors.extend(check_schema(df, required))
    n_checks += 1

    # Types
    numeric = ["pnl"]
    errors.extend(check_types(df, numeric))
    n_checks += 1

    # Nulls
    errors.extend(check_nulls(df, 0.20))
    n_checks += 1

    # Freshness
    errors.extend(check_freshness(df))
    n_checks += 1

    # Duplicates
    errors.extend(check_duplicates(df, ["entry_date", "exit_date"]))
    n_checks += 1

    # Empty check
    if df.empty:
        errors.append(StageError("EMPTY_DATA", "DataFrame is empty", "error",
                                  "Verify data source is populated."))
    n_checks += 1

    n_failed = sum(1 for e in errors if e.severity == "error")
    n_warn = sum(1 for e in errors if e.severity == "warning")
    elapsed = (time.monotonic() - t0) * 1000

    return StageResult(
        stage="data_ingestion", passed=n_failed == 0,
        elapsed_ms=elapsed, n_checks=n_checks,
        n_passed=n_checks - n_failed - n_warn,
        n_failed=n_failed, n_warnings=n_warn,
        errors=errors, input_shape=df.shape, output_shape=df.shape,
    )


def validate_features(df: pd.DataFrame) -> StageResult:
    """Stage 2: Validate feature engineering output."""
    t0 = time.monotonic()
    errors: List[StageError] = []
    n_checks = 0

    feature_cols = ["vix", "iv_rank", "regime", "rsi_14", "momentum_5d_pct"]
    present = [c for c in feature_cols if c in df.columns]
    if len(present) < 2:
        errors.append(StageError("FEW_FEATURES", f"Only {len(present)} feature columns found",
                                  "warning", "Run feature pipeline before validation."))
    n_checks += 1

    # Range checks
    if "vix" in df.columns:
        errors.extend(check_value_range(df["vix"], "vix", 5, 90))
        n_checks += 1
    if "iv_rank" in df.columns:
        errors.extend(check_value_range(df["iv_rank"], "iv_rank", 0, 100))
        n_checks += 1
    if "rsi_14" in df.columns:
        errors.extend(check_value_range(df["rsi_14"], "rsi_14", 0, 100))
        n_checks += 1

    # NaN ratio in features
    for col in present:
        null_pct = df[col].isna().mean() if col in df.columns else 0
        if null_pct > 0.30:
            errors.append(StageError("FEATURE_NULLS", f"'{col}' has {null_pct:.1%} nulls",
                                      "warning", f"Impute '{col}' in feature pipeline."))
        n_checks += 1

    n_failed = sum(1 for e in errors if e.severity == "error")
    n_warn = sum(1 for e in errors if e.severity == "warning")
    elapsed = (time.monotonic() - t0) * 1000

    return StageResult(
        stage="feature_engineering", passed=n_failed == 0,
        elapsed_ms=elapsed, n_checks=n_checks,
        n_passed=n_checks - n_failed - n_warn,
        n_failed=n_failed, n_warnings=n_warn,
        errors=errors, input_shape=df.shape, output_shape=df.shape,
    )


def validate_signals(
    df: pd.DataFrame, signal_col: str = "signal_score",
) -> StageResult:
    """Stage 3: Validate signal generation."""
    t0 = time.monotonic()
    errors: List[StageError] = []
    n_checks = 0

    if signal_col not in df.columns:
        errors.append(StageError("NO_SIGNAL", f"Missing '{signal_col}' column", "error",
                                  "Run signal generation before validation."))
        n_checks += 1
    else:
        # Bounds
        errors.extend(check_value_range(df[signal_col], signal_col, 0.0, 1.0))
        n_checks += 1

        # Distribution
        errors.extend(check_distribution(df[signal_col], signal_col, (0.2, 0.8)))
        n_checks += 1

        # Coverage (at least some trades scored)
        coverage = (df[signal_col] > 0).mean()
        if coverage < 0.5:
            errors.append(StageError("LOW_COVERAGE", f"Only {coverage:.1%} of trades have signal > 0",
                                      "warning", "Check signal generation for data gaps."))
        n_checks += 1

        # Not all same value
        if df[signal_col].nunique() < 3:
            errors.append(StageError("LOW_VARIANCE", "Signal has < 3 unique values",
                                      "warning", "Signal may be degenerate."))
        n_checks += 1

    n_failed = sum(1 for e in errors if e.severity == "error")
    n_warn = sum(1 for e in errors if e.severity == "warning")
    elapsed = (time.monotonic() - t0) * 1000

    return StageResult(
        stage="signal_generation", passed=n_failed == 0,
        elapsed_ms=elapsed, n_checks=n_checks,
        n_passed=n_checks - n_failed - n_warn,
        n_failed=n_failed, n_warnings=n_warn,
        errors=errors, input_shape=df.shape, output_shape=df.shape,
    )


def validate_model(
    df: pd.DataFrame,
    prediction_col: str = "model_pred",
    model_age_days: int = 0,
    max_model_age: int = 90,
) -> StageResult:
    """Stage 4: Validate model prediction."""
    t0 = time.monotonic()
    errors: List[StageError] = []
    n_checks = 0

    if prediction_col not in df.columns:
        # Not fatal — many pipelines don't have a separate model pred column
        errors.append(StageError("NO_MODEL_PRED", f"Missing '{prediction_col}' column",
                                  "info", "Add model predictions if using ML model."))
    else:
        errors.extend(check_value_range(df[prediction_col], prediction_col, 0.0, 1.0))
    n_checks += 1

    # Model staleness
    if model_age_days > max_model_age:
        errors.append(StageError("STALE_MODEL", f"Model is {model_age_days}d old (max {max_model_age})",
                                  "warning", "Retrain model with recent data."))
    n_checks += 1

    # Calibration: if predictions exist, check mean ≈ actual win rate
    if prediction_col in df.columns and "win" in df.columns:
        pred_mean = df[prediction_col].mean()
        actual_mean = df["win"].mean()
        if abs(pred_mean - actual_mean) > 0.15:
            errors.append(StageError("MISCALIBRATED",
                                      f"Pred mean={pred_mean:.3f} vs actual={actual_mean:.3f}",
                                      "warning", "Recalibrate model predictions."))
        n_checks += 1

    n_failed = sum(1 for e in errors if e.severity == "error")
    n_warn = sum(1 for e in errors if e.severity == "warning")
    elapsed = (time.monotonic() - t0) * 1000

    return StageResult(
        stage="model_prediction", passed=n_failed == 0,
        elapsed_ms=elapsed, n_checks=n_checks,
        n_passed=n_checks - n_failed - n_warn,
        n_failed=n_failed, n_warnings=n_warn,
        errors=errors, input_shape=df.shape, output_shape=df.shape,
    )


def validate_sizing(
    df: pd.DataFrame,
    contracts_col: str = "contracts",
    max_contracts: int = 50,
    max_notional: float = 50_000.0,
) -> StageResult:
    """Stage 5: Validate position sizing."""
    t0 = time.monotonic()
    errors: List[StageError] = []
    n_checks = 0

    if contracts_col in df.columns:
        errors.extend(check_value_range(df[contracts_col], contracts_col, 1, max_contracts))
        n_checks += 1

        # Notional check
        if "net_credit" in df.columns:
            notional = df[contracts_col] * df["net_credit"].abs() * 100
            over = (notional > max_notional).sum()
            if over > 0:
                errors.append(StageError("NOTIONAL_EXCEEDED",
                                          f"{over} trades exceed ${max_notional:,.0f} notional",
                                          "warning", "Reduce contract count or position limit."))
            n_checks += 1
    else:
        errors.append(StageError("NO_CONTRACTS", f"Missing '{contracts_col}' column",
                                  "info", "Add sizing output to pipeline."))
        n_checks += 1

    n_failed = sum(1 for e in errors if e.severity == "error")
    n_warn = sum(1 for e in errors if e.severity == "warning")
    elapsed = (time.monotonic() - t0) * 1000

    return StageResult(
        stage="position_sizing", passed=n_failed == 0,
        elapsed_ms=elapsed, n_checks=n_checks,
        n_passed=n_checks - n_failed - n_warn,
        n_failed=n_failed, n_warnings=n_warn,
        errors=errors, input_shape=df.shape, output_shape=df.shape,
    )


def validate_risk(
    df: pd.DataFrame,
    max_drawdown: float = 0.15,
    capital: float = 100_000.0,
) -> StageResult:
    """Stage 6: Validate risk checks."""
    t0 = time.monotonic()
    errors: List[StageError] = []
    n_checks = 0

    # Drawdown check
    if "pnl" in df.columns:
        cum_pnl = df["pnl"].cumsum()
        equity = capital + cum_pnl
        peak = equity.cummax()
        dd = (equity - peak) / peak.where(peak > 0, 1)
        worst_dd = abs(dd.min())
        if worst_dd > max_drawdown:
            errors.append(StageError("DRAWDOWN_EXCEEDED",
                                      f"Max drawdown {worst_dd:.1%} exceeds limit {max_drawdown:.1%}",
                                      "error", "Tighten risk gates or reduce position sizes."))
        n_checks += 1

    # Regime concentration
    if "regime" in df.columns:
        regime_counts = df["regime"].value_counts(normalize=True)
        for regime, pct in regime_counts.items():
            if pct > 0.8:
                errors.append(StageError("REGIME_CONCENTRATION",
                                          f"{pct:.1%} of trades in '{regime}' regime",
                                          "warning", "Diversify across regimes."))
        n_checks += 1

    # Consecutive losses
    if "win" in df.columns:
        wins = df["win"].values
        max_consec_loss = 0
        current = 0
        for w in wins:
            if w == 0:
                current += 1
                max_consec_loss = max(max_consec_loss, current)
            else:
                current = 0
        if max_consec_loss > 10:
            errors.append(StageError("CONSEC_LOSSES",
                                      f"{max_consec_loss} consecutive losses detected",
                                      "warning", "Review signal quality during loss streaks."))
        n_checks += 1

    n_failed = sum(1 for e in errors if e.severity == "error")
    n_warn = sum(1 for e in errors if e.severity == "warning")
    elapsed = (time.monotonic() - t0) * 1000

    return StageResult(
        stage="risk_checks", passed=n_failed == 0,
        elapsed_ms=elapsed, n_checks=n_checks,
        n_passed=n_checks - n_failed - n_warn,
        n_failed=n_failed, n_warnings=n_warn,
        errors=errors, input_shape=df.shape, output_shape=df.shape,
    )


def validate_orders(
    df: pd.DataFrame,
    side_col: str = "strategy_type",
) -> StageResult:
    """Stage 7: Validate order generation."""
    t0 = time.monotonic()
    errors: List[StageError] = []
    n_checks = 0

    # Valid sides/strategy types
    if side_col in df.columns:
        valid_types = {"CS", "IC", "SS", "credit_spread", "iron_condor", "straddle", "strangle", "unknown"}
        invalid = set(df[side_col].unique()) - valid_types
        if invalid:
            errors.append(StageError("INVALID_STRATEGY",
                                      f"Unknown strategy types: {invalid}",
                                      "warning", "Map to known strategy types."))
        n_checks += 1

    # Price sanity
    if "net_credit" in df.columns:
        credits = df["net_credit"].dropna()
        if len(credits) > 0:
            if (credits.abs() > 50).any():
                errors.append(StageError("PRICE_OUTLIER",
                                          "Net credit > $50 detected",
                                          "warning", "Verify pricing data."))
            n_checks += 1

    # Duplicate orders
    errors.extend(check_duplicates(df, ["entry_date", "exit_date", "strategy_type"]))
    n_checks += 1

    n_failed = sum(1 for e in errors if e.severity == "error")
    n_warn = sum(1 for e in errors if e.severity == "warning")
    elapsed = (time.monotonic() - t0) * 1000

    return StageResult(
        stage="order_generation", passed=n_failed == 0,
        elapsed_ms=elapsed, n_checks=n_checks,
        n_passed=n_checks - n_failed - n_warn,
        n_failed=n_failed, n_warnings=n_warn,
        errors=errors, input_shape=df.shape, output_shape=df.shape,
    )


def validate_hedge(
    df: pd.DataFrame,
    hedge_ratio_col: str = "hedge_ratio",
    max_hedge_ratio: float = 2.0,
) -> StageResult:
    """Stage 8: Validate hedge overlay."""
    t0 = time.monotonic()
    errors: List[StageError] = []
    n_checks = 0

    if hedge_ratio_col in df.columns:
        errors.extend(check_value_range(df[hedge_ratio_col], hedge_ratio_col, 0.0, max_hedge_ratio))
        n_checks += 1
    else:
        # Hedge overlay is optional
        errors.append(StageError("NO_HEDGE", "No hedge_ratio column — hedge overlay not active",
                                  "info", "Add hedge overlay if desired."))
        n_checks += 1

    n_failed = sum(1 for e in errors if e.severity == "error")
    n_warn = sum(1 for e in errors if e.severity == "warning")
    elapsed = (time.monotonic() - t0) * 1000

    return StageResult(
        stage="hedge_overlay", passed=n_failed == 0,
        elapsed_ms=elapsed, n_checks=n_checks,
        n_passed=n_checks - n_failed - n_warn,
        n_failed=n_failed, n_warnings=n_warn,
        errors=errors, input_shape=df.shape, output_shape=df.shape,
    )


# ── Core validator ───────────────────────────────────────────────────────


class PipelineValidator:
    """Production pipeline validator.

    Args:
        data: Trade DataFrame to validate.
        signal_col: Name of signal score column.
        prediction_col: Name of model prediction column.
        model_age_days: Age of current model in days.
        max_drawdown: Maximum allowed drawdown fraction.
        capital: Account capital for risk calculations.
        max_contracts: Maximum contracts per trade.
        max_notional: Maximum notional per trade.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signal_col: str = "signal_score",
        prediction_col: str = "model_pred",
        model_age_days: int = 0,
        max_drawdown: float = 0.15,
        capital: float = 100_000.0,
        max_contracts: int = 50,
        max_notional: float = 50_000.0,
    ):
        if data.empty:
            raise ValueError("data must not be empty")
        self.data = data.copy()
        self.signal_col = signal_col
        self.prediction_col = prediction_col
        self.model_age_days = model_age_days
        self.max_drawdown = max_drawdown
        self.capital = capital
        self.max_contracts = max_contracts
        self.max_notional = max_notional

    def validate(self) -> ValidationResult:
        """Run all validation stages."""
        t0 = time.monotonic()
        df = self.data

        stages: List[StageResult] = []
        stages.append(validate_data_ingestion(df))
        stages.append(validate_features(df))
        stages.append(validate_signals(df, self.signal_col))
        stages.append(validate_model(df, self.prediction_col, self.model_age_days))
        stages.append(validate_sizing(df, max_contracts=self.max_contracts,
                                       max_notional=self.max_notional))
        stages.append(validate_risk(df, self.max_drawdown, self.capital))
        stages.append(validate_orders(df))
        stages.append(validate_hedge(df))

        kill = evaluate_kill_switch(stages)

        total_errors = sum(s.n_failed for s in stages)
        total_warns = sum(s.n_warnings for s in stages)
        all_pass = all(s.passed for s in stages) and not kill.triggered
        elapsed = (time.monotonic() - t0) * 1000

        return ValidationResult(
            stages=stages,
            kill_switch=kill,
            overall_pass=all_pass,
            total_elapsed_ms=elapsed,
            n_stages=len(stages),
            n_stages_passed=sum(1 for s in stages if s.passed),
            n_total_errors=total_errors,
            n_total_warnings=total_warns,
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @staticmethod
    def generate_report(
        result: ValidationResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _build_html(result: ValidationResult) -> str:
    oc = "#3fb950" if result.overall_pass else "#f85149"
    ks = result.kill_switch
    ks_color = "#f85149" if ks.triggered else "#3fb950"
    ks_label = "TRIGGERED" if ks.triggered else "OK"

    # Stage rows
    stage_rows = ""
    for s in result.stages:
        sc = "#3fb950" if s.passed else "#f85149"
        icon = "&#10003;" if s.passed else "&#10007;"
        err_list = ""
        if s.errors:
            items = "".join(
                f"<li><code>[{e.severity.upper()}]</code> {e.message}<br/>"
                f"<em>{e.remediation}</em></li>"
                for e in s.errors
            )
            err_list = f"<ul class='err-list'>{items}</ul>"

        stage_rows += f"""<tr>
          <td style="text-align:left"><span style="color:{sc}">{icon}</span> {s.stage}</td>
          <td>{s.n_checks}</td>
          <td style="color:#3fb950">{s.n_passed}</td>
          <td style="color:#f85149">{s.n_failed}</td>
          <td style="color:#d29922">{s.n_warnings}</td>
          <td>{s.elapsed_ms:.1f}ms</td>
          <td style="text-align:left;font-size:0.85em">{s.input_shape}</td>
        </tr>"""
        if err_list:
            stage_rows += f'<tr><td colspan="7" style="text-align:left;padding-left:30px">{err_list}</td></tr>'

    # Kill switch checks
    ks_rows = ""
    for name, passed, detail in ks.checks:
        kc = "#3fb950" if passed else "#f85149"
        ki = "&#10003;" if passed else "&#10007;"
        ks_rows += f"<tr><td style='text-align:left'><span style='color:{kc}'>{ki}</span> {name}</td><td style='text-align:left'>{detail}</td></tr>"

    # Pipeline flow diagram
    def _stage_span(s: StageResult) -> str:
        c = "#3fb950" if s.passed else "#f85149"
        return f"<span style='color:{c}'>{s.stage}</span>"
    flow = " &rarr; ".join(_stage_span(s) for s in result.stages)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><title>Pipeline Validation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
  h1,h2,h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; }}
  .hero {{ background: #161b22; border: 2px solid {oc}; border-radius: 12px;
           padding: 24px; text-align: center; margin: 20px 0; }}
  .hero .big {{ font-size: 2.5em; font-weight: 800; color: {oc}; }}
  .hero .sub {{ color: #8b949e; }}
  .flow {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; font-size: 0.95em; text-align: center; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 10px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.1em; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.dt th, table.dt td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; }}
  ul.err-list {{ margin: 4px 0; padding-left: 20px; font-size: 0.85em; }}
  ul.err-list li {{ margin: 4px 0; }}
  ul.err-list em {{ color: #8b949e; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }}
</style>
</head>
<body>
<h1>Pipeline Validation Report</h1>

<div class="hero">
  <div class="big">{"PIPELINE VALID" if result.overall_pass else "VALIDATION FAILED"}</div>
  <div class="sub">{result.timestamp} &middot; {result.total_elapsed_ms:.1f}ms &middot;
     {result.n_stages_passed}/{result.n_stages} stages passed</div>
</div>

<div class="flow"><strong>Pipeline Flow:</strong> {flow}</div>

<div class="summary">
  <div class="stat"><div class="label">Stages</div><div class="value">{result.n_stages_passed}/{result.n_stages}</div></div>
  <div class="stat"><div class="label">Errors</div><div class="value" style="color:#f85149">{result.n_total_errors}</div></div>
  <div class="stat"><div class="label">Warnings</div><div class="value" style="color:#d29922">{result.n_total_warnings}</div></div>
  <div class="stat"><div class="label">Kill Switch</div><div class="value" style="color:{ks_color}">{ks_label}</div></div>
  <div class="stat"><div class="label">Total Time</div><div class="value">{result.total_elapsed_ms:.1f}ms</div></div>
</div>

<h2>Stage Results</h2>
<table class="dt">
  <tr><th style="text-align:left">Stage</th><th>Checks</th><th>Pass</th><th>Fail</th><th>Warn</th><th>Time</th><th style="text-align:left">Shape</th></tr>
  {stage_rows}
</table>

<h2>Kill Switch</h2>
<div class="card">
  <p>Status: <strong style="color:{ks_color}">{ks_label}</strong>
     {f' &mdash; {ks.reason}' if ks.reason else ''}</p>
  <table class="dt"><tr><th style="text-align:left">Check</th><th style="text-align:left">Detail</th></tr>{ks_rows}</table>
</div>

</body></html>"""
