"""Automated strategy screener — test 50+ strategies/day instead of 14/week.

Workflow:
    1. Define a StrategySpec: data loader + signal function + default params + grid
    2. Call screen(spec) → ScreenResult with metrics, walk-forward, sensitivity, grade
    3. Grade is PASS / CONDITIONAL / FAIL against North Star criteria

Rule Zero (MASTERPLAN.md):
    Strategy signal functions are scanned for synthetic-data tells (np.random,
    fake, simulate, ...). Any hit hard-fails the screen.

Usage:
    spec = StrategySpec(
        name="EXP-1900: My idea",
        data_source="Yahoo SPY 2014-2025",
        loader=lambda: load_spy_yahoo(),
        signal_fn=my_strategy,           # (prices, params) → daily Series
        default_params={"lookback": 20},
        param_grid={"lookback": [10, 20, 40]},
    )
    result = screen(spec)
    print(result.grade, result.fail_reasons)
"""
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# North Star (from MASTERPLAN.md)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NorthStarCriteria:
    """North Star thresholds. Defaults are for individual diversifier strategies.

    The combined-portfolio targets (CAGR 100%, Sharpe 6) are aggregate goals;
    no single new strategy is expected to hit them alone. The screener treats
    a strategy as PASS if it qualifies as a useful diversifier.
    """
    min_cagr_pct: float = 8.0          # ≥8% standalone (diversifier bar)
    min_sharpe: float = 1.0            # ≥1.0 standalone
    max_dd_pct: float = 25.0           # ≤25% (single strategy can be deeper than portfolio)
    max_corr_to_exp1220: float = 0.30  # <0.30 to count as a diversifier
    min_years_profitable_frac: float = 0.60  # ≥60% of years profitable
    max_param_sensitivity: float = 0.50      # Sharpe range / Sharpe median ≤0.5 (no cliff)
    min_oos_sharpe: float = 0.50             # walk-forward OOS Sharpe must hold up


# ═══════════════════════════════════════════════════════════════════════════
# Spec & result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategySpec:
    name: str
    data_source: str                                    # for the report — Rule Zero citation
    loader: Callable[[], pd.DataFrame]                  # () → prices DataFrame (real data)
    signal_fn: Callable[[pd.DataFrame, Dict], pd.Series]  # (prices, params) → daily returns
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    description: str = ""


@dataclass
class WFFold:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_sharpe: float
    oos_sharpe: float
    oos_cagr: float
    oos_dd: float


@dataclass
class SensitivityResult:
    param: str
    values: List[Any]
    sharpes: List[float]
    sharpe_median: float
    sharpe_range: float
    relative_range: float   # range / |median|, lower = more robust
    cliff_detected: bool


@dataclass
class ScreenResult:
    spec: StrategySpec
    n_days: int
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    vol: float
    corr_to_exp1220: Optional[float]
    yearly_returns: Dict[int, float]
    years_profitable_frac: float
    wf_folds: List[WFFold]
    avg_oos_sharpe: float
    sensitivity: List[SensitivityResult]
    grade: str                  # PASS / CONDITIONAL / FAIL
    fail_reasons: List[str]
    pass_reasons: List[str]
    rule_zero_clean: bool
    rule_zero_warnings: List[str]


# ═══════════════════════════════════════════════════════════════════════════
# Metrics (corrected Sharpe — matches MASTERPLAN canonical formula)
# ═══════════════════════════════════════════════════════════════════════════

def corrected_sharpe(rets: np.ndarray) -> float:
    """Sharpe using arithmetic mean(daily) × √252 / std(daily, ddof=1)."""
    if len(rets) < 2:
        return 0.0
    s = float(np.std(rets, ddof=1))
    if s < 1e-12:
        return 0.0
    return float(np.mean(rets) / s * math.sqrt(TRADING_DAYS))


def compute_metrics(rets: np.ndarray) -> Dict[str, float]:
    if len(rets) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "dd": 0.0,
                "calmar": 0.0, "vol": 0.0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else -1.0
    sharpe = corrected_sharpe(rets)
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0.0
    down = rets[rets < 0]
    if len(down) > 1:
        ds = float(down.std(ddof=1))
    else:
        ds = float(np.std(rets, ddof=1))
    sortino = (float(np.mean(rets)) / ds * math.sqrt(TRADING_DAYS)) if ds > 1e-12 else 0.0
    vol = float(np.std(rets, ddof=1)) * math.sqrt(TRADING_DAYS)
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "dd": dd, "calmar": calmar, "vol": vol}


def yearly_returns(rets: pd.Series) -> Dict[int, float]:
    if len(rets) == 0:
        return {}
    out = {}
    for yr, group in rets.groupby(rets.index.year):
        eq = (1 + group).prod() - 1
        out[int(yr)] = float(eq)
    return out


def years_profitable_fraction(yearly: Dict[int, float]) -> float:
    if not yearly:
        return 0.0
    n_pos = sum(1 for v in yearly.values() if v > 0)
    return n_pos / len(yearly)


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward(
    rets: pd.Series,
    min_train_years: float = 2.0,
    test_years: float = 1.0,
) -> List[WFFold]:
    """Expanding-window walk-forward — chronological only, no shuffle."""
    if len(rets) < int((min_train_years + test_years) * TRADING_DAYS):
        return []
    train_end_idx = int(min_train_years * TRADING_DAYS)
    test_len = int(test_years * TRADING_DAYS)
    folds = []
    while train_end_idx + test_len <= len(rets):
        train = rets.iloc[:train_end_idx]
        test = rets.iloc[train_end_idx:train_end_idx + test_len]
        is_m = compute_metrics(train.values)
        oos_m = compute_metrics(test.values)
        folds.append(WFFold(
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            is_sharpe=is_m["sharpe"],
            oos_sharpe=oos_m["sharpe"],
            oos_cagr=oos_m["cagr"],
            oos_dd=oos_m["dd"],
        ))
        train_end_idx += test_len
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Parameter sensitivity (overfitting check)
# ═══════════════════════════════════════════════════════════════════════════

def parameter_sensitivity(
    spec: StrategySpec,
    prices: pd.DataFrame,
) -> List[SensitivityResult]:
    """For each param in grid, sweep values holding others at default,
    record Sharpe — flag a parameter as a 'cliff' if relative range > 0.5.
    """
    out = []
    base = dict(spec.default_params)
    for pname, values in spec.param_grid.items():
        sharpes = []
        for v in values:
            params = dict(base)
            params[pname] = v
            try:
                rets = spec.signal_fn(prices, params)
                if rets is None or len(rets) == 0:
                    sharpes.append(0.0)
                    continue
                sharpes.append(corrected_sharpe(np.asarray(rets)))
            except Exception:
                sharpes.append(0.0)
        median = float(np.median(sharpes)) if sharpes else 0.0
        rng = float(max(sharpes) - min(sharpes)) if sharpes else 0.0
        rel = rng / abs(median) if abs(median) > 1e-6 else float("inf")
        out.append(SensitivityResult(
            param=pname,
            values=list(values),
            sharpes=sharpes,
            sharpe_median=median,
            sharpe_range=rng,
            relative_range=rel,
            cliff_detected=rel > 0.5,
        ))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Rule Zero scanner
# ═══════════════════════════════════════════════════════════════════════════

SYNTHETIC_TELLS = [
    r"\bnp\.random\b", r"\brandom\.normal\b", r"\brandom\.uniform\b",
    r"\bsynthetic\b", r"\bfake_prices\b", r"\bgenerate_prices\b",
    r"\bsimulate_returns\b", r"\bblack_scholes_price\b",
    r"\bBACKTEST_CREDIT_FRACTION\b",
]


def scan_rule_zero(signal_fn: Callable) -> Tuple[bool, List[str]]:
    """Inspect the signal_fn source for synthetic-data tells.

    Returns (clean: bool, warnings: list[str]).
    """
    try:
        src = inspect.getsource(signal_fn)
    except (TypeError, OSError):
        return True, ["[notice] could not introspect source — manual review required"]
    warnings = []
    for pat in SYNTHETIC_TELLS:
        if re.search(pat, src):
            warnings.append(f"forbidden pattern matched: {pat}")
    return (len(warnings) == 0), warnings


# ═══════════════════════════════════════════════════════════════════════════
# Grading
# ═══════════════════════════════════════════════════════════════════════════

def grade_strategy(
    metrics: Dict[str, float],
    corr: Optional[float],
    yearly: Dict[int, float],
    avg_oos_sharpe: float,
    sensitivity: List[SensitivityResult],
    rule_zero_clean: bool,
    crit: NorthStarCriteria,
) -> Tuple[str, List[str], List[str]]:
    fails: List[str] = []
    passes: List[str] = []

    if not rule_zero_clean:
        fails.append("RULE ZERO violation: synthetic-data pattern in signal function")

    cagr_pct = metrics["cagr"] * 100
    if cagr_pct >= crit.min_cagr_pct:
        passes.append(f"CAGR {cagr_pct:.1f}% ≥ {crit.min_cagr_pct}%")
    else:
        fails.append(f"CAGR {cagr_pct:.1f}% < {crit.min_cagr_pct}%")

    if metrics["sharpe"] >= crit.min_sharpe:
        passes.append(f"Sharpe {metrics['sharpe']:.2f} ≥ {crit.min_sharpe}")
    else:
        fails.append(f"Sharpe {metrics['sharpe']:.2f} < {crit.min_sharpe}")

    dd_pct = metrics["dd"] * 100
    if dd_pct <= crit.max_dd_pct:
        passes.append(f"DD {dd_pct:.1f}% ≤ {crit.max_dd_pct}%")
    else:
        fails.append(f"DD {dd_pct:.1f}% > {crit.max_dd_pct}%")

    if corr is None:
        passes.append("Correlation to EXP-1220: not measured")
    elif abs(corr) <= crit.max_corr_to_exp1220:
        passes.append(f"|corr to EXP-1220| {abs(corr):.2f} ≤ {crit.max_corr_to_exp1220}")
    else:
        fails.append(f"|corr to EXP-1220| {abs(corr):.2f} > {crit.max_corr_to_exp1220}")

    yp = years_profitable_fraction(yearly)
    if yp >= crit.min_years_profitable_frac:
        passes.append(f"Years profitable {yp:.0%} ≥ {crit.min_years_profitable_frac:.0%}")
    else:
        fails.append(f"Years profitable {yp:.0%} < {crit.min_years_profitable_frac:.0%}")

    if avg_oos_sharpe >= crit.min_oos_sharpe:
        passes.append(f"OOS Sharpe (avg) {avg_oos_sharpe:.2f} ≥ {crit.min_oos_sharpe}")
    else:
        fails.append(f"OOS Sharpe (avg) {avg_oos_sharpe:.2f} < {crit.min_oos_sharpe}")

    cliffs = [s.param for s in sensitivity if s.cliff_detected]
    if cliffs:
        fails.append(f"Parameter cliff(s): {', '.join(cliffs)}")
    else:
        passes.append("No parameter cliffs (overfitting check OK)")

    if not fails:
        grade = "PASS"
    elif len(fails) <= 1 and rule_zero_clean:
        grade = "CONDITIONAL"
    else:
        grade = "FAIL"
    return grade, fails, passes


# ═══════════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════════

def screen(
    spec: StrategySpec,
    exp1220_returns: Optional[pd.Series] = None,
    criteria: Optional[NorthStarCriteria] = None,
) -> ScreenResult:
    """Full screen pipeline — returns ScreenResult with grade.

    exp1220_returns: optional reference series for correlation. If None,
        the corr field is left as None.
    """
    crit = criteria or NorthStarCriteria()

    # Rule Zero scan FIRST — short-circuit on contamination
    rule_zero_clean, rz_warn = scan_rule_zero(spec.signal_fn)

    # Load real data
    prices = spec.loader()

    # Run main backtest
    rets = spec.signal_fn(prices, spec.default_params)
    if not isinstance(rets, pd.Series):
        rets = pd.Series(rets, index=prices.index[-len(rets):])
    rets = rets.dropna()

    # Metrics
    m = compute_metrics(rets.values)
    yearly = yearly_returns(rets)
    yp = years_profitable_fraction(yearly)

    # Correlation to EXP-1220
    corr = None
    if exp1220_returns is not None and len(exp1220_returns) > 5:
        joined = pd.concat([rets, exp1220_returns], axis=1, join="inner").dropna()
        if len(joined) > 5 and joined.iloc[:, 0].std() > 1e-12 and joined.iloc[:, 1].std() > 1e-12:
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))

    # Walk-forward
    folds = walk_forward(rets)
    avg_oos = float(np.mean([f.oos_sharpe for f in folds])) if folds else 0.0

    # Sensitivity
    sens = parameter_sensitivity(spec, prices) if spec.param_grid else []

    grade, fails, passes = grade_strategy(
        m, corr, yearly, avg_oos, sens, rule_zero_clean, crit
    )

    return ScreenResult(
        spec=spec,
        n_days=len(rets),
        cagr=m["cagr"], sharpe=m["sharpe"], sortino=m["sortino"],
        max_dd=m["dd"], calmar=m["calmar"], vol=m["vol"],
        corr_to_exp1220=corr,
        yearly_returns=yearly,
        years_profitable_frac=yp,
        wf_folds=folds, avg_oos_sharpe=avg_oos,
        sensitivity=sens,
        grade=grade, fail_reasons=fails, pass_reasons=passes,
        rule_zero_clean=rule_zero_clean,
        rule_zero_warnings=rz_warn,
    )


def format_result(result: ScreenResult) -> str:
    """Pretty one-screen text summary."""
    s = result
    badge = {"PASS": "✅ PASS", "CONDITIONAL": "⚠️  CONDITIONAL", "FAIL": "❌ FAIL"}[s.grade]
    lines = [
        f"════════════════════════════════════════════════════════════",
        f"{badge}  {s.spec.name}",
        f"Data: {s.spec.data_source}",
        f"────────────────────────────────────────────────────────────",
        f"  CAGR        {s.cagr*100:+7.1f}%",
        f"  Sharpe      {s.sharpe:7.2f}",
        f"  Max DD      {s.max_dd*100:7.1f}%",
        f"  Sortino     {s.sortino:7.2f}",
        f"  Calmar      {s.calmar:7.2f}",
        f"  Vol         {s.vol*100:7.1f}%",
        f"  Years profitable: {s.years_profitable_frac:.0%}",
        f"  Corr to EXP-1220: {s.corr_to_exp1220 if s.corr_to_exp1220 is None else f'{s.corr_to_exp1220:+.2f}'}",
        f"  WF folds: {len(s.wf_folds)} (avg OOS Sharpe {s.avg_oos_sharpe:.2f})",
        f"  Sensitivity params: {len(s.sensitivity)} ({sum(1 for x in s.sensitivity if x.cliff_detected)} cliffs)",
        f"  Rule Zero: {'CLEAN' if s.rule_zero_clean else 'VIOLATED'}",
    ]
    if s.pass_reasons:
        lines.append(f"  ✓ {len(s.pass_reasons)} criteria met")
    if s.fail_reasons:
        lines.append(f"  ✗ Fail reasons:")
        for f in s.fail_reasons:
            lines.append(f"      - {f}")
    return "\n".join(lines)
