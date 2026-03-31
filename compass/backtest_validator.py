"""Backtest validation suite – detects common backtesting pitfalls, overfitting,
and statistical anomalies to ensure all backtests are trustworthy.

Checks:
  1. Look-ahead bias detection (future data leaking into decisions)
  2. Survivorship bias indicators
  3. Data snooping / multiple testing adjustments
  4. Unrealistic fill assumptions
  5. Missing or insufficient transaction costs
  6. Statistical tests: runs test, Ljung-Box, Kolmogorov-Smirnov
  7. Overfitting: IS vs OOS degradation, parameter cliffs, min backtest length
  8. Composite validation score 0-100
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Grade constants ─────────────────────────────────────────────────────────
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# ── Thresholds ──────────────────────────────────────────────────────────────
MIN_BACKTEST_DAYS = 252          # 1 year minimum
MIN_TRADES = 30                  # statistical minimum
MAX_REALISTIC_SHARPE = 5.0       # above this = likely overfit
FANTASY_SHARPE = 10.0            # impossible
MAX_REALISTIC_WIN_RATE = 0.90    # >90% suspicious for credit spreads
MAX_FILL_RATE = 1.0              # 100% fill rate suspicious
MIN_COST_BPS = 0.5               # minimum realistic transaction cost
AUTOCORR_PVALUE_THRESHOLD = 0.05
RUNS_PVALUE_THRESHOLD = 0.05
OOS_DEGRADATION_WARN = 0.30      # 30% degradation is a warning
OOS_DEGRADATION_FAIL = 0.60      # 60% degradation is a failure
CLIFF_THRESHOLD = 0.50           # 50% performance drop = cliff


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    """Outcome of a single validation check."""
    name: str
    grade: str          # PASS, WARN, FAIL
    message: str
    score: float        # 0-100 contribution
    category: str       # bias, statistical, overfitting, realism
    details: str = ""


@dataclass
class StatTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    pvalue: float
    passed: bool
    interpretation: str


@dataclass
class OverfitMetrics:
    """In-sample vs out-of-sample comparison."""
    is_sharpe: float
    oos_sharpe: float
    sharpe_degradation: float   # (IS - OOS) / IS
    is_win_rate: float
    oos_win_rate: float
    is_avg_pnl: float
    oos_avg_pnl: float
    pnl_degradation: float
    parameter_cliff: bool
    min_length_met: bool


@dataclass
class ValidationResult:
    """Complete backtest validation output."""
    score: float = 0.0                           # composite 0-100
    grade: str = ""                              # PASS / WARN / FAIL
    checks: List[CheckResult] = field(default_factory=list)
    stat_tests: List[StatTestResult] = field(default_factory=list)
    overfit: Optional[OverfitMetrics] = None
    recommendations: List[str] = field(default_factory=list)
    n_trades: int = 0
    n_days: int = 0
    generated_at: str = ""


# ── Statistical helpers ─────────────────────────────────────────────────────
def _runs_test(series: np.ndarray) -> Tuple[float, float]:
    """Wald–Wolfowitz runs test for randomness.

    Returns (z_statistic, p_value).
    """
    n = len(series)
    if n < 10:
        return (0.0, 1.0)

    median = np.median(series)
    binary = (series >= median).astype(int)

    n1 = int(np.sum(binary))
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return (0.0, 1.0)

    # Count runs
    runs = 1 + int(np.sum(binary[1:] != binary[:-1]))

    expected = 1 + 2 * n0 * n1 / n
    var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
    if var <= 0:
        return (0.0, 1.0)

    z = (runs - expected) / math.sqrt(var)
    # Two-tailed p-value from standard normal
    p = 2.0 * _norm_sf(abs(z))
    return (float(z), float(p))


def _ljung_box(series: np.ndarray, lags: int = 10) -> Tuple[float, float]:
    """Ljung–Box test for autocorrelation.

    Returns (Q_statistic, p_value).
    """
    n = len(series)
    if n < lags + 5:
        return (0.0, 1.0)

    mean = np.mean(series)
    centered = series - mean
    var = np.sum(centered ** 2) / n
    if var < 1e-15:
        return (0.0, 1.0)

    q = 0.0
    for k in range(1, lags + 1):
        rk = np.sum(centered[k:] * centered[:-k]) / (n * var)
        q += rk ** 2 / (n - k)
    q *= n * (n + 2)

    # Chi-squared p-value (degrees of freedom = lags)
    p = _chi2_sf(q, lags)
    return (float(q), float(p))


def _ks_test_normal(series: np.ndarray) -> Tuple[float, float]:
    """One-sample Kolmogorov–Smirnov test against normal distribution.

    Returns (D_statistic, p_value).
    """
    n = len(series)
    if n < 5:
        return (0.0, 1.0)

    mean = np.mean(series)
    std = np.std(series, ddof=1)
    if std < 1e-15:
        return (0.0, 1.0)

    sorted_s = np.sort(series)
    empirical = np.arange(1, n + 1) / n
    theoretical = _norm_cdf((sorted_s - mean) / std)
    d = float(np.max(np.abs(empirical - theoretical)))

    # Asymptotic p-value
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    if lam <= 0:
        return (d, 1.0)
    p = 2.0 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam) for k in range(1, 20))
    return (d, float(np.clip(p, 0.0, 1.0)))


def _norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + _erf(np.asarray(x) / np.sqrt(2)))


def _norm_sf(x: float) -> float:
    """Standard normal survival function P(X > x)."""
    return float(1.0 - _norm_cdf(x))


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun approximation of error function."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
           t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))


def _chi2_sf(x: float, df: int) -> float:
    """Chi-squared survival function approximation via Wilson-Hilferty."""
    if df <= 0 or x <= 0:
        return 1.0
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return float(_norm_sf(z))


# ── Core validator ──────────────────────────────────────────────────────────
class BacktestValidator:
    """Validates backtest results for common pitfalls and overfitting."""

    def __init__(
        self,
        min_days: int = MIN_BACKTEST_DAYS,
        min_trades: int = MIN_TRADES,
        cost_bps: float = MIN_COST_BPS,
    ) -> None:
        self.min_days = min_days
        self.min_trades = min_trades
        self.cost_bps = cost_bps

    # ── Public API ──────────────────────────────────────────────────────────
    def validate(
        self,
        trades: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        oos_trades: Optional[pd.DataFrame] = None,
        oos_returns: Optional[pd.Series] = None,
        param_sweep: Optional[Dict[str, List[float]]] = None,
        n_params_tested: int = 1,
    ) -> ValidationResult:
        """Run full validation suite.

        Parameters
        ----------
        trades : pd.DataFrame
            Must have columns: date (or entry_date), pnl.
            Optional: fill_price, slippage, commission, exit_reason.
        returns : pd.Series, optional
            Daily strategy returns for statistical tests.
        oos_trades : pd.DataFrame, optional
            Out-of-sample trades for overfitting detection.
        oos_returns : pd.Series, optional
            Out-of-sample returns.
        param_sweep : dict, optional
            Mapping param_name → list of performance values (e.g. Sharpe)
            at different parameter settings, for cliff detection.
        n_params_tested : int
            Number of parameter combinations tested (for data snooping adjustment).
        """
        trades = self._normalize(trades)
        if trades.empty:
            return ValidationResult(
                score=0, grade=FAIL, generated_at=self._now(),
                recommendations=["No valid trades to validate."],
            )

        checks: List[CheckResult] = []
        stat_tests: List[StatTestResult] = []

        # Bias checks
        checks.append(self._check_look_ahead(trades))
        checks.append(self._check_survivorship(trades))
        checks.append(self._check_data_snooping(n_params_tested))
        checks.append(self._check_fill_realism(trades))
        checks.append(self._check_transaction_costs(trades))

        # Realism checks
        checks.append(self._check_win_rate(trades))
        checks.append(self._check_sharpe(trades, returns))
        checks.append(self._check_min_length(trades))
        checks.append(self._check_min_trades(trades))

        # Statistical tests
        if returns is not None and len(returns) > 20:
            ret_arr = returns.dropna().values
            stat_tests.extend(self._run_stat_tests(ret_arr))

        # Overfitting
        overfit = None
        if oos_trades is not None:
            oos_trades = self._normalize(oos_trades)
            overfit = self._compute_overfit(trades, oos_trades, returns, oos_returns)
            checks.append(self._check_oos_degradation(overfit))

        if param_sweep:
            checks.append(self._check_param_cliff(param_sweep))

        # Score
        score = self._compute_score(checks, stat_tests)
        grade = PASS if score >= 70 else WARN if score >= 40 else FAIL
        recs = self._generate_recommendations(checks, stat_tests, overfit)

        n_days = 0
        if "date" in trades.columns:
            dates = pd.to_datetime(trades["date"])
            n_days = (dates.max() - dates.min()).days

        return ValidationResult(
            score=score,
            grade=grade,
            checks=checks,
            stat_tests=stat_tests,
            overfit=overfit,
            recommendations=recs,
            n_trades=len(trades),
            n_days=n_days,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: ValidationResult,
        output_path: str | Path = "reports/backtest_validation.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Backtest validation report written to %s", path)
        return path

    # ── Normalization ───────────────────────────────────────────────────────
    @staticmethod
    def _normalize(trades: pd.DataFrame) -> pd.DataFrame:
        df = trades.copy()
        if "entry_date" in df.columns and "date" not in df.columns:
            df["date"] = df["entry_date"]
        if "date" not in df.columns or "pnl" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    # ── Bias checks ─────────────────────────────────────────────────────────
    @staticmethod
    def _check_look_ahead(trades: pd.DataFrame) -> CheckResult:
        """Detect look-ahead bias: exit before entry, or suspiciously perfect
        timing around known events."""
        if "exit_date" in trades.columns:
            entry = pd.to_datetime(trades["date"])
            exit_d = pd.to_datetime(trades["exit_date"])
            violations = int((exit_d < entry).sum())
            if violations > 0:
                return CheckResult(
                    name="Look-Ahead Bias", grade=FAIL,
                    message=f"{violations} trades have exit before entry",
                    score=0, category="bias",
                )

        # Check for perfect win streaks at the start (often a sign of
        # using future data to select entry)
        if len(trades) >= 20:
            first_20 = trades.head(20)
            if (first_20["pnl"] > 0).all():
                return CheckResult(
                    name="Look-Ahead Bias", grade=WARN,
                    message="First 20 trades are all winners (suspicious)",
                    score=60, category="bias",
                )

        return CheckResult(
            name="Look-Ahead Bias", grade=PASS,
            message="No look-ahead bias detected",
            score=100, category="bias",
        )

    @staticmethod
    def _check_survivorship(trades: pd.DataFrame) -> CheckResult:
        """Check for survivorship bias indicators."""
        if "ticker" in trades.columns:
            tickers = trades["ticker"].nunique()
            if tickers == 1:
                return CheckResult(
                    name="Survivorship Bias", grade=WARN,
                    message="Single ticker tested — may have survivorship bias",
                    score=60, category="bias",
                )
        # Check for gaps in dates (missing data periods)
        dates = pd.to_datetime(trades["date"])
        if len(dates) > 10:
            diffs = dates.diff().dropna()
            max_gap = diffs.max().days
            if max_gap > 30:
                return CheckResult(
                    name="Survivorship Bias", grade=WARN,
                    message=f"Max gap of {max_gap} days — possible data gap",
                    score=70, category="bias",
                    details=f"Largest gap: {max_gap} calendar days",
                )

        return CheckResult(
            name="Survivorship Bias", grade=PASS,
            message="No survivorship bias indicators",
            score=100, category="bias",
        )

    @staticmethod
    def _check_data_snooping(n_params: int) -> CheckResult:
        """Bonferroni-style adjustment for multiple parameter testing."""
        if n_params <= 1:
            return CheckResult(
                name="Data Snooping", grade=PASS,
                message="Single configuration tested",
                score=100, category="bias",
            )
        # Adjusted significance level
        adj_alpha = 0.05 / n_params
        if n_params > 100:
            return CheckResult(
                name="Data Snooping", grade=FAIL,
                message=f"{n_params} params tested — high snooping risk (adj α={adj_alpha:.6f})",
                score=20, category="bias",
            )
        if n_params > 20:
            return CheckResult(
                name="Data Snooping", grade=WARN,
                message=f"{n_params} params tested — moderate snooping risk (adj α={adj_alpha:.4f})",
                score=55, category="bias",
            )
        return CheckResult(
            name="Data Snooping", grade=PASS,
            message=f"{n_params} params tested — acceptable (adj α={adj_alpha:.4f})",
            score=90, category="bias",
        )

    @staticmethod
    def _check_fill_realism(trades: pd.DataFrame) -> CheckResult:
        """Check for unrealistic fill assumptions."""
        if "slippage" in trades.columns or "slippage_applied" in trades.columns:
            slip_col = "slippage" if "slippage" in trades.columns else "slippage_applied"
            zero_slip = (trades[slip_col] == 0).mean()
            if zero_slip > 0.95:
                return CheckResult(
                    name="Fill Realism", grade=WARN,
                    message=f"{zero_slip:.0%} of trades have zero slippage",
                    score=50, category="realism",
                )
            return CheckResult(
                name="Fill Realism", grade=PASS,
                message="Slippage applied to trades",
                score=100, category="realism",
            )

        return CheckResult(
            name="Fill Realism", grade=WARN,
            message="No slippage data — fill realism unknown",
            score=60, category="realism",
        )

    def _check_transaction_costs(self, trades: pd.DataFrame) -> CheckResult:
        """Check whether transaction costs are accounted for."""
        if "commission" in trades.columns:
            total_comm = trades["commission"].sum()
            total_pnl = trades["pnl"].sum()
            if total_pnl != 0:
                cost_ratio = abs(total_comm / total_pnl)
                if cost_ratio < 0.001:
                    return CheckResult(
                        name="Transaction Costs", grade=WARN,
                        message=f"Commissions are {cost_ratio:.2%} of P&L — suspiciously low",
                        score=50, category="realism",
                    )
            return CheckResult(
                name="Transaction Costs", grade=PASS,
                message="Commissions included in trade data",
                score=100, category="realism",
            )

        # Estimate if pnl seems to ignore costs
        if len(trades) > 0:
            avg_pnl = trades["pnl"].mean()
            if abs(avg_pnl) < self.cost_bps * 0.01:
                return CheckResult(
                    name="Transaction Costs", grade=WARN,
                    message="Avg P&L near cost threshold — costs may not be included",
                    score=55, category="realism",
                )

        return CheckResult(
            name="Transaction Costs", grade=WARN,
            message="No commission column — cannot verify cost inclusion",
            score=60, category="realism",
        )

    # ── Realism checks ─────────────────────────────────────────────────────
    @staticmethod
    def _check_win_rate(trades: pd.DataFrame) -> CheckResult:
        wins = (trades["pnl"] > 0).mean()
        if wins >= 0.95:
            return CheckResult(
                name="Win Rate Realism", grade=FAIL,
                message=f"Win rate {wins:.1%} — unrealistic",
                score=10, category="realism",
            )
        if wins >= MAX_REALISTIC_WIN_RATE:
            return CheckResult(
                name="Win Rate Realism", grade=WARN,
                message=f"Win rate {wins:.1%} — aggressive",
                score=55, category="realism",
            )
        return CheckResult(
            name="Win Rate Realism", grade=PASS,
            message=f"Win rate {wins:.1%}",
            score=100, category="realism",
        )

    @staticmethod
    def _check_sharpe(trades: pd.DataFrame, returns: Optional[pd.Series]) -> CheckResult:
        if returns is not None and len(returns) > 20:
            ret = returns.dropna()
            mean_r = float(ret.mean())
            std_r = float(ret.std())
            if std_r > 1e-12:
                sharpe = mean_r / std_r * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            # Estimate from trade PnL
            pnls = trades["pnl"]
            if pnls.std() > 1e-12:
                sharpe = float(pnls.mean() / pnls.std()) * np.sqrt(len(pnls))
            else:
                sharpe = 0.0

        if abs(sharpe) >= FANTASY_SHARPE:
            return CheckResult(
                name="Sharpe Realism", grade=FAIL,
                message=f"Sharpe {sharpe:.2f} — impossible",
                score=0, category="realism",
            )
        if abs(sharpe) >= MAX_REALISTIC_SHARPE:
            return CheckResult(
                name="Sharpe Realism", grade=WARN,
                message=f"Sharpe {sharpe:.2f} — likely overfit",
                score=40, category="realism",
            )
        return CheckResult(
            name="Sharpe Realism", grade=PASS,
            message=f"Sharpe {sharpe:.2f}",
            score=100, category="realism",
        )

    def _check_min_length(self, trades: pd.DataFrame) -> CheckResult:
        dates = pd.to_datetime(trades["date"])
        n_days = (dates.max() - dates.min()).days
        if n_days < self.min_days:
            return CheckResult(
                name="Minimum Length", grade=FAIL,
                message=f"{n_days} days — need {self.min_days} minimum",
                score=max(0, int(n_days / self.min_days * 70)),
                category="overfitting",
            )
        return CheckResult(
            name="Minimum Length", grade=PASS,
            message=f"{n_days} days of data",
            score=100, category="overfitting",
        )

    def _check_min_trades(self, trades: pd.DataFrame) -> CheckResult:
        n = len(trades)
        if n < self.min_trades:
            return CheckResult(
                name="Minimum Trades", grade=FAIL,
                message=f"{n} trades — need {self.min_trades} minimum",
                score=max(0, int(n / self.min_trades * 70)),
                category="overfitting",
            )
        return CheckResult(
            name="Minimum Trades", grade=PASS,
            message=f"{n} trades",
            score=100, category="overfitting",
        )

    # ── Statistical tests ───────────────────────────────────────────────────
    def _run_stat_tests(self, returns: np.ndarray) -> List[StatTestResult]:
        results: List[StatTestResult] = []

        # Runs test
        z, p = _runs_test(returns)
        passed = p > RUNS_PVALUE_THRESHOLD
        results.append(StatTestResult(
            test_name="Runs Test (Randomness)",
            statistic=z, pvalue=p, passed=passed,
            interpretation="Returns appear random" if passed
            else "Returns show non-random patterns (possible curve fitting)",
        ))

        # Ljung-Box
        q, p = _ljung_box(returns)
        passed = p > AUTOCORR_PVALUE_THRESHOLD
        results.append(StatTestResult(
            test_name="Ljung-Box (Autocorrelation)",
            statistic=q, pvalue=p, passed=passed,
            interpretation="No significant autocorrelation" if passed
            else "Significant autocorrelation detected",
        ))

        # KS test
        d, p = _ks_test_normal(returns)
        results.append(StatTestResult(
            test_name="KS Test (Normality)",
            statistic=d, pvalue=p, passed=p > 0.05,
            interpretation="Returns approximately normal" if p > 0.05
            else "Returns deviate from normal (expected for credit spreads)",
        ))

        return results

    # ── Overfitting detection ───────────────────────────────────────────────
    @staticmethod
    def _compute_overfit(
        is_trades: pd.DataFrame,
        oos_trades: pd.DataFrame,
        is_returns: Optional[pd.Series],
        oos_returns: Optional[pd.Series],
    ) -> OverfitMetrics:
        is_pnl = is_trades["pnl"]
        oos_pnl = oos_trades["pnl"] if not oos_trades.empty else pd.Series(dtype=float)

        is_wr = float((is_pnl > 0).mean()) if len(is_pnl) > 0 else 0.0
        oos_wr = float((oos_pnl > 0).mean()) if len(oos_pnl) > 0 else 0.0
        is_avg = float(is_pnl.mean()) if len(is_pnl) > 0 else 0.0
        oos_avg = float(oos_pnl.mean()) if len(oos_pnl) > 0 else 0.0

        def _sharpe(ret: Optional[pd.Series]) -> float:
            if ret is None or len(ret) < 5:
                return 0.0
            r = ret.dropna()
            s = float(r.std())
            return float(r.mean()) / s * np.sqrt(252) if s > 1e-12 else 0.0

        is_sharpe = _sharpe(is_returns)
        oos_sharpe = _sharpe(oos_returns)

        s_deg = (is_sharpe - oos_sharpe) / is_sharpe if abs(is_sharpe) > 1e-9 else 0.0
        p_deg = (is_avg - oos_avg) / abs(is_avg) if abs(is_avg) > 1e-9 else 0.0

        return OverfitMetrics(
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            sharpe_degradation=s_deg,
            is_win_rate=is_wr,
            oos_win_rate=oos_wr,
            is_avg_pnl=is_avg,
            oos_avg_pnl=oos_avg,
            pnl_degradation=p_deg,
            parameter_cliff=False,
            min_length_met=len(oos_trades) >= 20,
        )

    @staticmethod
    def _check_oos_degradation(om: OverfitMetrics) -> CheckResult:
        deg = om.sharpe_degradation
        if deg >= OOS_DEGRADATION_FAIL:
            return CheckResult(
                name="OOS Degradation", grade=FAIL,
                message=f"Sharpe degrades {deg:.0%} out-of-sample",
                score=10, category="overfitting",
            )
        if deg >= OOS_DEGRADATION_WARN:
            return CheckResult(
                name="OOS Degradation", grade=WARN,
                message=f"Sharpe degrades {deg:.0%} out-of-sample",
                score=50, category="overfitting",
            )
        return CheckResult(
            name="OOS Degradation", grade=PASS,
            message=f"OOS Sharpe degradation {deg:.0%} — acceptable",
            score=100, category="overfitting",
        )

    @staticmethod
    def _check_param_cliff(param_sweep: Dict[str, List[float]]) -> CheckResult:
        """Detect parameter cliffs: sharp performance drops near optimal."""
        for param_name, values in param_sweep.items():
            if len(values) < 3:
                continue
            arr = np.array(values, dtype=float)
            best_idx = int(np.argmax(arr))
            best_val = arr[best_idx]
            if abs(best_val) < 1e-12:
                continue

            # Check neighbors
            for offset in [-1, 1]:
                neighbor = best_idx + offset
                if 0 <= neighbor < len(arr):
                    drop = (best_val - arr[neighbor]) / abs(best_val)
                    if drop >= CLIFF_THRESHOLD:
                        return CheckResult(
                            name="Parameter Sensitivity", grade=FAIL,
                            message=f"Cliff in '{param_name}': {drop:.0%} drop at neighbor",
                            score=20, category="overfitting",
                        )

        return CheckResult(
            name="Parameter Sensitivity", grade=PASS,
            message="No parameter cliffs detected",
            score=100, category="overfitting",
        )

    # ── Scoring ─────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_score(
        checks: List[CheckResult], stat_tests: List[StatTestResult],
    ) -> float:
        if not checks:
            return 0.0
        check_score = sum(c.score for c in checks) / len(checks)

        if stat_tests:
            stat_score = sum(100.0 if t.passed else 40.0 for t in stat_tests) / len(stat_tests)
            return check_score * 0.7 + stat_score * 0.3
        return check_score

    # ── Recommendations ─────────────────────────────────────────────────────
    @staticmethod
    def _generate_recommendations(
        checks: List[CheckResult],
        stat_tests: List[StatTestResult],
        overfit: Optional[OverfitMetrics],
    ) -> List[str]:
        recs: List[str] = []
        for c in checks:
            if c.grade == FAIL:
                recs.append(f"[CRITICAL] {c.name}: {c.message}")
            elif c.grade == WARN:
                recs.append(f"[WARNING] {c.name}: {c.message}")

        for t in stat_tests:
            if not t.passed:
                recs.append(f"[STAT] {t.test_name}: {t.interpretation} (p={t.pvalue:.4f})")

        if overfit:
            if overfit.sharpe_degradation > OOS_DEGRADATION_WARN:
                recs.append(
                    f"[OVERFIT] Consider reducing model complexity — "
                    f"IS Sharpe={overfit.is_sharpe:.2f} vs OOS={overfit.oos_sharpe:.2f}"
                )
            if overfit.parameter_cliff:
                recs.append("[OVERFIT] Parameter cliff detected — widen parameter ranges")

        return recs

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: ValidationResult) -> str:
        cards = self._html_cards(r)
        checklist = self._html_checklist(r.checks)
        stat_section = self._html_stat_tests(r.stat_tests)
        overfit_section = self._html_overfit(r.overfit)
        recs = self._html_recommendations(r.recommendations)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Backtest Validation</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pass{{color:#4ade80}}.warn{{color:#fbbf24}}.fail{{color:#f87171}}
.score-big{{font-size:2.5rem;font-weight:800}}
.rec{{background:#1e293b;border-left:3px solid #f87171;padding:10px 14px;margin-bottom:8px;border-radius:4px;font-size:.85rem}}
.rec.warning{{border-color:#fbbf24}}
.rec.stat{{border-color:#a78bfa}}
</style>
</head>
<body>
<h1>Backtest Validation Report</h1>
<p class="sub">Generated {r.generated_at or 'N/A'}</p>

{cards}
{checklist}
{stat_section}
{overfit_section}
{recs}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: ValidationResult) -> str:
        score_cls = "pass" if r.score >= 70 else "warn" if r.score >= 40 else "fail"
        n_pass = sum(1 for c in r.checks if c.grade == PASS)
        n_warn = sum(1 for c in r.checks if c.grade == WARN)
        n_fail = sum(1 for c in r.checks if c.grade == FAIL)
        return f"""<div class="grid">
<div class="card"><div class="lbl">Validation Score</div><div class="val score-big {score_cls}">{r.score:.0f}</div></div>
<div class="card"><div class="lbl">Grade</div><div class="val {score_cls}">{r.grade}</div></div>
<div class="card"><div class="lbl">Trades</div><div class="val">{r.n_trades}</div></div>
<div class="card"><div class="lbl">Days</div><div class="val">{r.n_days}</div></div>
<div class="card"><div class="lbl">Checks Passed</div><div class="val pass">{n_pass}</div></div>
<div class="card"><div class="lbl">Warnings</div><div class="val warn">{n_warn}</div></div>
<div class="card"><div class="lbl">Failures</div><div class="val fail">{n_fail}</div></div>
</div>"""

    @staticmethod
    def _html_checklist(checks: List[CheckResult]) -> str:
        if not checks:
            return ""
        rows = ""
        for c in checks:
            grade_cls = c.grade.lower()
            rows += (
                f'<tr><td class="{grade_cls}">{c.grade}</td>'
                f"<td>{c.name}</td>"
                f"<td>{c.category}</td>"
                f"<td>{c.message}</td>"
                f"<td>{c.score:.0f}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Validation Checklist</h2>
<table>
<thead><tr><th>Status</th><th>Check</th><th>Category</th><th>Detail</th><th>Score</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_stat_tests(tests: List[StatTestResult]) -> str:
        if not tests:
            return ""
        rows = ""
        for t in tests:
            cls = "pass" if t.passed else "fail"
            rows += (
                f'<tr><td>{t.test_name}</td>'
                f"<td>{t.statistic:.4f}</td>"
                f"<td>{t.pvalue:.4f}</td>"
                f'<td class="{cls}">{"PASS" if t.passed else "FAIL"}</td>'
                f"<td>{t.interpretation}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Statistical Tests</h2>
<table>
<thead><tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Result</th><th>Interpretation</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_overfit(om: Optional[OverfitMetrics]) -> str:
        if om is None:
            return ""
        s_cls = "fail" if om.sharpe_degradation > OOS_DEGRADATION_FAIL else "warn" if om.sharpe_degradation > OOS_DEGRADATION_WARN else "pass"
        return f"""<div class="sec">
<h2>Overfitting Analysis</h2>
<table>
<thead><tr><th>Metric</th><th>In-Sample</th><th>Out-of-Sample</th><th>Degradation</th></tr></thead>
<tbody>
<tr><td>Sharpe Ratio</td><td>{om.is_sharpe:.2f}</td><td>{om.oos_sharpe:.2f}</td><td class="{s_cls}">{om.sharpe_degradation:.0%}</td></tr>
<tr><td>Win Rate</td><td>{om.is_win_rate:.1%}</td><td>{om.oos_win_rate:.1%}</td><td>{om.is_win_rate - om.oos_win_rate:.1%}</td></tr>
<tr><td>Avg P&L</td><td>{om.is_avg_pnl:.2f}</td><td>{om.oos_avg_pnl:.2f}</td><td>{om.pnl_degradation:.0%}</td></tr>
<tr><td>Min OOS Length</td><td colspan="2"></td><td class="{'pass' if om.min_length_met else 'fail'}">{'Met' if om.min_length_met else 'Not met'}</td></tr>
<tr><td>Param Cliff</td><td colspan="2"></td><td class="{'fail' if om.parameter_cliff else 'pass'}">{'Detected' if om.parameter_cliff else 'None'}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_recommendations(recs: List[str]) -> str:
        if not recs:
            return '<div class="sec"><h2>Recommendations</h2><p class="pass">All checks passed — no actions needed.</p></div>'
        items = ""
        for r in recs:
            if "[CRITICAL]" in r:
                cls = "rec"
            elif "[STAT]" in r:
                cls = "rec stat"
            else:
                cls = "rec warning"
            items += f'<div class="{cls}">{r}</div>'
        return f"""<div class="sec">
<h2>Recommendations</h2>
{items}
</div>"""
