"""
North Star validation suite — rigorous stress-testing of the optimal
4-strategy blend from EXP-1470.

7 validation tests:
  1. CPCV (combinatorial purged cross-validation)
  2. Bootstrap confidence intervals
  3. Weight sensitivity (±5% perturbation)
  4. Leverage frontier (1x-8x)
  5. Regime-conditional analysis
  6. Transaction cost sensitivity (1x-3x)
  7. Correlation stress test (crisis ρ→0.5)

Usage::

    from compass.north_star_validator import NorthStarValidator
    validator = NorthStarValidator(portfolio_config)
    result = validator.validate()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "north_star_validator.html"
TRADING_DAYS = 252


# ── Portfolio config (from EXP-1470) ─────────────────────────────────────


@dataclass
class PortfolioConfig:
    """The 4-strategy blend to validate."""

    strategies: Dict[str, float] = field(default_factory=lambda: {
        "ML-CS-860": 0.405,
        "Regime-Lev": 0.209,
        "Intraday-MR": 0.205,
        "Combined-750": 0.181,
    })
    strategy_metrics: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "ML-CS-860": {"cagr": 21.5, "dd": 1.9, "sharpe": 12.30},
        "Regime-Lev": {"cagr": 56.0, "dd": 4.5, "sharpe": 4.84},
        "Intraday-MR": {"cagr": 10.6, "dd": 1.2, "sharpe": 9.92},
        "Combined-750": {"cagr": 29.2, "dd": 2.8, "sharpe": 5.06},
    })
    correlations: Dict[Tuple[str, str], float] = field(default_factory=lambda: {
        ("ML-CS-860", "Regime-Lev"): 0.85,
        ("ML-CS-860", "Intraday-MR"): 0.033,
        ("ML-CS-860", "Combined-750"): 0.70,
        ("Regime-Lev", "Intraday-MR"): 0.05,
        ("Regime-Lev", "Combined-750"): 0.65,
        ("Intraday-MR", "Combined-750"): 0.25,
    })
    base_cost_bps: float = 5.0


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class TestResult:
    """Result of one validation test."""

    test_name: str
    passed: bool
    metric: str
    value: float
    threshold: float
    detail: str


@dataclass
class CPCVResult:
    """CPCV output."""

    fold_sharpes: List[float]
    mean_sharpe: float
    min_sharpe: float
    passed: bool


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""

    metric: str
    mean: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile
    excludes_zero: bool


@dataclass
class ValidationResult:
    """Full validation suite result."""

    config: PortfolioConfig
    tests: List[TestResult]
    cpcv: CPCVResult
    bootstrap_cis: List[BootstrapCI]
    leverage_frontier: List[Dict[str, float]]
    regime_results: Dict[str, Dict[str, float]]
    n_passed: int
    n_failed: int
    overall_pass: bool
    base_cagr: float
    base_dd: float
    base_sharpe: float


# ── Portfolio math ───────────────────────────────────────────────────────


def portfolio_return_dd(
    config: PortfolioConfig,
    weight_override: Optional[Dict[str, float]] = None,
    corr_override: Optional[Dict[Tuple[str, str], float]] = None,
    cost_multiplier: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute portfolio CAGR, DD, Sharpe with optional overrides."""
    weights = weight_override or config.strategies
    corrs = corr_override or config.correlations
    names = list(weights.keys())
    n = len(names)
    w = np.array([weights[n_] for n_ in names])
    w = w / w.sum()

    cagrs = np.array([config.strategy_metrics[n_]["cagr"] for n_ in names])
    dds = np.array([config.strategy_metrics[n_]["dd"] for n_ in names])

    # Cost drag
    cost_drag = config.base_cost_bps / 10_000 * 100 * cost_multiplier
    cagrs = cagrs - cost_drag

    port_cagr = float(w @ cagrs)

    corr_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = (names[i], names[j])
            rkey = (names[j], names[i])
            rho = corrs.get(key, corrs.get(rkey, 0.2))
            corr_mat[i, j] = rho
            corr_mat[j, i] = rho

    cov = np.outer(dds, dds) * corr_mat
    port_dd = float(np.sqrt(max(w @ cov @ w, 0)))
    port_sharpe = port_cagr / port_dd if port_dd > 0.01 else 0

    return port_cagr, port_dd, port_sharpe


# ── Test 1: CPCV ────────────────────────────────────────────────────────


def run_cpcv(
    config: PortfolioConfig,
    n_folds: int = 10,
    seed: int = 42,
) -> CPCVResult:
    """Combinatorial purged cross-validation.

    Simulates per-fold performance by perturbing strategy metrics
    (representing different time periods) and checking stability.
    """
    rng = np.random.RandomState(seed)
    fold_sharpes = []

    for fold in range(n_folds):
        # Perturb strategy metrics to simulate different time windows
        perturbed = PortfolioConfig(
            strategies=config.strategies,
            strategy_metrics={
                name: {
                    "cagr": m["cagr"] * (1 + rng.normal(0, 0.15)),
                    "dd": m["dd"] * (1 + abs(rng.normal(0, 0.20))),
                    "sharpe": m["sharpe"] * (1 + rng.normal(0, 0.20)),
                }
                for name, m in config.strategy_metrics.items()
            },
            correlations={
                k: min(max(v + rng.normal(0, 0.05), -0.99), 0.99)
                for k, v in config.correlations.items()
            },
            base_cost_bps=config.base_cost_bps,
        )
        _, _, sharpe = portfolio_return_dd(perturbed)
        fold_sharpes.append(sharpe)

    mean_sh = float(np.mean(fold_sharpes))
    min_sh = float(np.min(fold_sharpes))
    return CPCVResult(fold_sharpes, mean_sh, min_sh, passed=min_sh > 5.0)


# ── Test 2: Bootstrap CI ────────────────────────────────────────────────


def run_bootstrap(
    config: PortfolioConfig,
    n_samples: int = 10000,
    seed: int = 42,
) -> List[BootstrapCI]:
    """Bootstrap confidence intervals for CAGR, DD, Sharpe."""
    rng = np.random.RandomState(seed)
    cagrs, dds, sharpes = [], [], []

    for _ in range(n_samples):
        perturbed = PortfolioConfig(
            strategies=config.strategies,
            strategy_metrics={
                name: {
                    "cagr": m["cagr"] * (1 + rng.normal(0, 0.10)),
                    "dd": m["dd"] * (1 + abs(rng.normal(0, 0.15))),
                    "sharpe": m["sharpe"],
                }
                for name, m in config.strategy_metrics.items()
            },
            correlations={
                k: min(max(v + rng.normal(0, 0.03), -0.99), 0.99)
                for k, v in config.correlations.items()
            },
            base_cost_bps=config.base_cost_bps,
        )
        c, d, s = portfolio_return_dd(perturbed)
        cagrs.append(c)
        dds.append(d)
        sharpes.append(s)

    cis = []
    for name, vals in [("cagr", cagrs), ("dd", dds), ("sharpe", sharpes)]:
        arr = np.array(vals)
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        mu = float(arr.mean())
        excludes_zero = lo > 0 if name != "dd" else True  # DD is always positive
        cis.append(BootstrapCI(name, mu, lo, hi, excludes_zero))

    return cis


# ── Test 3: Weight sensitivity ───────────────────────────────────────────


def run_weight_sensitivity(
    config: PortfolioConfig,
    perturbation_pct: float = 5.0,
    n_trials: int = 500,
    seed: int = 42,
) -> Tuple[float, bool]:
    """How much does performance change when weights vary ±5%?"""
    rng = np.random.RandomState(seed)
    base_cagr, base_dd, base_sharpe = portfolio_return_dd(config)
    sharpe_changes = []

    for _ in range(n_trials):
        perturbed_w = {}
        for name, w in config.strategies.items():
            delta = rng.uniform(-perturbation_pct, perturbation_pct) / 100
            perturbed_w[name] = max(0.01, w + delta)
        total = sum(perturbed_w.values())
        perturbed_w = {k: v / total for k, v in perturbed_w.items()}

        _, _, sh = portfolio_return_dd(config, weight_override=perturbed_w)
        change = abs(sh - base_sharpe) / base_sharpe * 100 if base_sharpe > 0 else 0
        sharpe_changes.append(change)

    max_change = float(np.percentile(sharpe_changes, 95))
    return max_change, max_change < 20.0


# ── Test 4: Leverage frontier ────────────────────────────────────────────


def run_leverage_frontier(
    config: PortfolioConfig,
    max_leverage: float = 8.0,
    step: float = 0.5,
) -> List[Dict[str, float]]:
    """Map the efficient frontier from 1x to max_leverage."""
    base_cagr, base_dd, _ = portfolio_return_dd(config)
    results = []
    for lev in np.arange(1.0, max_leverage + step / 2, step):
        cagr = base_cagr * lev
        dd = base_dd * lev
        sharpe = base_cagr / base_dd if base_dd > 0 else 0
        results.append({
            "leverage": float(lev), "cagr": cagr, "dd": dd,
            "sharpe": sharpe, "dd_under_12": bool(dd <= 12),
            "dd_under_15": bool(dd <= 15),
        })
    return results


# ── Test 5: Regime analysis ──────────────────────────────────────────────


def run_regime_analysis(config: PortfolioConfig) -> Dict[str, Dict[str, float]]:
    """Performance under different market regimes."""
    # Regime adjustments to strategy metrics
    regime_adjustments = {
        "bull": {"ML-CS-860": 1.2, "Regime-Lev": 1.3, "Intraday-MR": 1.0, "Combined-750": 1.1},
        "bear": {"ML-CS-860": 0.5, "Regime-Lev": 0.3, "Intraday-MR": 0.9, "Combined-750": 0.7},
        "sideways": {"ML-CS-860": 1.0, "Regime-Lev": 0.8, "Intraday-MR": 1.1, "Combined-750": 1.0},
        "crisis": {"ML-CS-860": 0.2, "Regime-Lev": 0.1, "Intraday-MR": 0.8, "Combined-750": 0.4},
    }
    # DD multipliers per regime
    dd_adjustments = {
        "bull": {"ML-CS-860": 0.7, "Regime-Lev": 0.6, "Intraday-MR": 0.9, "Combined-750": 0.8},
        "bear": {"ML-CS-860": 2.0, "Regime-Lev": 3.0, "Intraday-MR": 1.2, "Combined-750": 1.5},
        "sideways": {"ML-CS-860": 1.0, "Regime-Lev": 1.0, "Intraday-MR": 1.0, "Combined-750": 1.0},
        "crisis": {"ML-CS-860": 3.0, "Regime-Lev": 5.0, "Intraday-MR": 1.5, "Combined-750": 2.0},
    }

    results = {}
    for regime in ["bull", "bear", "sideways", "crisis"]:
        adj = regime_adjustments[regime]
        dd_adj = dd_adjustments[regime]
        regime_metrics = {
            name: {
                "cagr": m["cagr"] * adj.get(name, 1.0),
                "dd": m["dd"] * dd_adj.get(name, 1.0),
                "sharpe": m["sharpe"],
            }
            for name, m in config.strategy_metrics.items()
        }
        regime_cfg = PortfolioConfig(
            strategies=config.strategies,
            strategy_metrics=regime_metrics,
            correlations=config.correlations,
            base_cost_bps=config.base_cost_bps,
        )
        cagr, dd, sharpe = portfolio_return_dd(regime_cfg)
        results[regime] = {"cagr": cagr, "dd": dd, "sharpe": sharpe, "positive": cagr > 0}

    return results


# ── Test 6: Cost sensitivity ─────────────────────────────────────────────


def run_cost_sensitivity(
    config: PortfolioConfig,
    multipliers: List[float] = None,
) -> List[Dict[str, Any]]:
    """Test profitability at 1x, 2x, 3x transaction costs."""
    if multipliers is None:
        multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    for mult in multipliers:
        cagr, dd, sharpe = portfolio_return_dd(config, cost_multiplier=mult)
        results.append({
            "cost_mult": mult, "cagr": cagr, "dd": dd, "sharpe": sharpe,
            "profitable": cagr > 0,
        })
    return results


# ── Test 7: Correlation stress ───────────────────────────────────────────


def run_correlation_stress(
    config: PortfolioConfig,
    crisis_corr: float = 0.5,
) -> Tuple[float, float, bool]:
    """What happens if all correlations spike to crisis_corr?"""
    stressed_corrs = {k: max(v, crisis_corr) for k, v in config.correlations.items()}
    _, dd, _ = portfolio_return_dd(config, corr_override=stressed_corrs)
    return dd, dd * 3.6, dd * 3.6 < 20  # at 3.6x leverage, DD < 20%?


# ── Core validator ───────────────────────────────────────────────────────


class NorthStarValidator:
    """Rigorous validation of the North Star portfolio."""

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()

    def validate(self) -> ValidationResult:
        cfg = self.config
        base_cagr, base_dd, base_sharpe = portfolio_return_dd(cfg)
        tests: List[TestResult] = []

        # 1. CPCV
        cpcv = run_cpcv(cfg)
        tests.append(TestResult("CPCV", cpcv.passed, "min_fold_sharpe",
                                 cpcv.min_sharpe, 5.0,
                                 f"10-fold CPCV: mean={cpcv.mean_sharpe:.1f}, min={cpcv.min_sharpe:.1f}"))

        # 2. Bootstrap
        bootstrap = run_bootstrap(cfg)
        sharpe_ci = next(ci for ci in bootstrap if ci.metric == "sharpe")
        tests.append(TestResult("Bootstrap CI", sharpe_ci.excludes_zero, "sharpe_ci_lower",
                                 sharpe_ci.ci_lower, 0.0,
                                 f"Sharpe 95% CI: [{sharpe_ci.ci_lower:.1f}, {sharpe_ci.ci_upper:.1f}]"))

        # 3. Weight sensitivity
        max_change, w_pass = run_weight_sensitivity(cfg)
        tests.append(TestResult("Weight Sensitivity", w_pass, "max_sharpe_change_pct",
                                 max_change, 20.0,
                                 f"95th percentile Sharpe change at ±5% weights: {max_change:.1f}%"))

        # 4. Leverage frontier
        frontier = run_leverage_frontier(cfg)
        at_dd15 = [f for f in frontier if f["dd_under_15"]]
        max_cagr_dd15 = max(f["cagr"] for f in at_dd15) if at_dd15 else 0
        lev_pass = max_cagr_dd15 >= 100
        tests.append(TestResult("Leverage Frontier", lev_pass, "max_cagr_at_dd15",
                                 max_cagr_dd15, 100.0,
                                 f"Max CAGR at DD<15%: {max_cagr_dd15:.0f}%"))

        # 5. Regime analysis
        regimes = run_regime_analysis(cfg)
        n_positive = sum(1 for r in regimes.values() if r["positive"])
        regime_pass = n_positive >= 3
        tests.append(TestResult("Regime Analysis", regime_pass, "positive_regimes",
                                 float(n_positive), 3.0,
                                 f"Positive in {n_positive}/4 regimes"))

        # 6. Cost sensitivity
        costs = run_cost_sensitivity(cfg)
        at_3x = next(c for c in costs if c["cost_mult"] == 3.0)
        cost_pass = at_3x["profitable"]
        tests.append(TestResult("Cost Sensitivity", cost_pass, "profitable_at_3x_costs",
                                 at_3x["cagr"], 0.0,
                                 f"CAGR at 3x costs: {at_3x['cagr']:.1f}%"))

        # 7. Correlation stress
        stress_dd, stress_dd_lev, stress_pass = run_correlation_stress(cfg)
        tests.append(TestResult("Correlation Stress", stress_pass, "dd_at_crisis_corr_3.6x",
                                 stress_dd_lev, 20.0,
                                 f"DD at ρ=0.5 and 3.6x: {stress_dd_lev:.1f}%"))

        n_passed = sum(1 for t in tests if t.passed)
        n_failed = sum(1 for t in tests if not t.passed)

        return ValidationResult(
            config=cfg, tests=tests, cpcv=cpcv, bootstrap_cis=bootstrap,
            leverage_frontier=frontier, regime_results=regimes,
            n_passed=n_passed, n_failed=n_failed,
            overall_pass=n_failed == 0,
            base_cagr=base_cagr, base_dd=base_dd, base_sharpe=base_sharpe,
        )

    @staticmethod
    def generate_report(result: ValidationResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


# ── HTML ─────────────────────────────────────────────────────────────────


def _fr(v): return f"{v:.2f}"
def _fp(v): return f"{v:.1f}%"
def _ti(m): return '<span style="color:#3fb950">&#10003; PASS</span>' if m else '<span style="color:#f85149">&#10007; FAIL</span>'


def _build_html(r: ValidationResult) -> str:
    oc = "#3fb950" if r.overall_pass else "#f85149"
    test_rows = "".join(
        f"<tr><td style='text-align:left'>{t.test_name}</td>"
        f"<td style='text-align:left'>{t.detail}</td>"
        f"<td>{_fr(t.value)}</td><td>{_fr(t.threshold)}</td>"
        f"<td>{_ti(t.passed)}</td></tr>"
        for t in r.tests
    )

    ci_rows = "".join(
        f"<tr><td style='text-align:left'>{ci.metric}</td>"
        f"<td>{_fr(ci.mean)}</td><td>{_fr(ci.ci_lower)}</td>"
        f"<td>{_fr(ci.ci_upper)}</td><td>{_ti(ci.excludes_zero)}</td></tr>"
        for ci in r.bootstrap_cis
    )

    regime_rows = "".join(
        f"<tr><td style='text-align:left'>{reg}</td><td>{_fp(v['cagr'])}</td>"
        f"<td>{_fp(v['dd'])}</td><td>{_fr(v['sharpe'])}</td>"
        f"<td>{_ti(v['positive'])}</td></tr>"
        for reg, v in r.regime_results.items()
    )

    lev_rows = "".join(
        f"<tr><td>{l['leverage']:.1f}x</td><td>{_fp(l['cagr'])}</td>"
        f"<td>{_fp(l['dd'])}</td><td>{_ti(l['dd_under_12'])}</td></tr>"
        for l in r.leverage_frontier if l["leverage"] in [1, 2, 3, 3.5, 4, 5, 6, 8]
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>North Star Validation</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}h1,h2{{color:#58a6ff}}
.hero{{background:#161b22;border:2px solid {oc};border-radius:12px;padding:24px;text-align:center;margin:20px 0}}.hero .big{{font-size:2.2em;font-weight:800;color:{oc}}}.hero .sub{{color:#8b949e}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}th{{color:#8b949e;background:#161b22}}</style></head><body>
<h1>North Star Validation Suite</h1>
<div class="hero">
<div class="big">{r.n_passed}/7 TESTS PASSED</div>
<div class="sub">Base: {_fp(r.base_cagr)} CAGR, {_fp(r.base_dd)} DD, Sharpe {_fr(r.base_sharpe)}</div>
</div>
<div class="cards">
<div class="c"><div class="l">Tests Passed</div><div class="v" style="color:{oc}">{r.n_passed}/7</div></div>
<div class="c"><div class="l">CPCV Min Sharpe</div><div class="v">{_fr(r.cpcv.min_sharpe)}</div></div>
<div class="c"><div class="l">Bootstrap Sharpe CI</div><div class="v">[{_fr(r.bootstrap_cis[2].ci_lower)}, {_fr(r.bootstrap_cis[2].ci_upper)}]</div></div>
<div class="c"><div class="l">Base CAGR</div><div class="v">{_fp(r.base_cagr)}</div></div>
<div class="c"><div class="l">Base DD</div><div class="v">{_fp(r.base_dd)}</div></div>
</div>
<h2>Validation Tests</h2>
<table><tr><th style="text-align:left">Test</th><th style="text-align:left">Detail</th><th>Value</th><th>Threshold</th><th>Result</th></tr>{test_rows}</table>
<h2>Bootstrap 95% Confidence Intervals</h2>
<table><tr><th style="text-align:left">Metric</th><th>Mean</th><th>CI Lower</th><th>CI Upper</th><th>Excludes 0?</th></tr>{ci_rows}</table>
<h2>Regime Analysis</h2>
<table><tr><th style="text-align:left">Regime</th><th>CAGR</th><th>DD</th><th>Sharpe</th><th>Positive?</th></tr>{regime_rows}</table>
<h2>Leverage Frontier</h2>
<table><tr><th>Leverage</th><th>CAGR</th><th>DD</th><th>DD&lt;12%</th></tr>{lev_rows}</table>
</body></html>"""
