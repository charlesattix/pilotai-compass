"""Bayesian configuration optimizer – Gaussian process surrogate model for
experiment parameter tuning with acquisition functions, multi-objective
optimization, and convergence analysis.

Provides:
  1. GP surrogate model for objective functions (Sharpe, returns, drawdown)
  2. Acquisition functions: UCB, EI, PI
  3. Parameter space: continuous, discrete, categorical
  4. Optimization history with convergence tracking
  5. Multi-objective optimization (Sharpe + drawdown) with Pareto frontier
  6. Warm-start from previous experiment results
  7. HTML report with convergence plot, param sensitivity, Pareto frontier
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ── Parameter space ─────────────────────────────────────────────────────────
class ParamType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass
class ParamDef:
    """Definition of a single tuneable parameter."""
    name: str
    param_type: str   # continuous / discrete / categorical
    low: float = 0.0
    high: float = 1.0
    choices: List[Any] = field(default_factory=list)  # for categorical/discrete
    default: Any = None

    def sample(self, rng: np.random.RandomState) -> Any:
        if self.param_type == ParamType.CONTINUOUS:
            return float(rng.uniform(self.low, self.high))
        if self.param_type == ParamType.DISCRETE:
            if self.choices:
                return self.choices[rng.randint(len(self.choices))]
            return int(rng.randint(int(self.low), int(self.high) + 1))
        if self.param_type == ParamType.CATEGORICAL:
            return self.choices[rng.randint(len(self.choices))]
        return self.default

    def encode(self, value: Any) -> float:
        """Encode a value to numeric for GP input."""
        if self.param_type == ParamType.CATEGORICAL:
            return float(self.choices.index(value)) if value in self.choices else 0.0
        if self.param_type == ParamType.DISCRETE and self.choices:
            return float(self.choices.index(value)) if value in self.choices else 0.0
        return float(value)

    def decode(self, encoded: float) -> Any:
        if self.param_type == ParamType.CATEGORICAL:
            idx = int(round(max(0, min(encoded, len(self.choices) - 1))))
            return self.choices[idx]
        if self.param_type == ParamType.DISCRETE:
            if self.choices:
                idx = int(round(max(0, min(encoded, len(self.choices) - 1))))
                return self.choices[idx]
            return int(round(max(self.low, min(encoded, self.high))))
        return float(np.clip(encoded, self.low, self.high))


# ── Acquisition functions ───────────────────────────────────────────────────
class AcquisitionFunc(str, Enum):
    UCB = "ucb"
    EI = "ei"
    PI = "pi"


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def _erf(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
           t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))


def compute_acquisition(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    func: str = AcquisitionFunc.UCB,
    kappa: float = 2.0,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute acquisition function values.

    Parameters
    ----------
    mu : array — GP mean predictions
    sigma : array — GP std predictions
    best_y : float — best observed objective so far
    func : str — "ucb", "ei", or "pi"
    kappa : float — exploration parameter for UCB
    xi : float — improvement threshold for EI/PI
    """
    sigma = np.maximum(sigma, 1e-9)

    if func == AcquisitionFunc.UCB:
        return mu + kappa * sigma

    z = (mu - best_y - xi) / sigma

    if func == AcquisitionFunc.EI:
        return (mu - best_y - xi) * _norm_cdf(z) + sigma * _norm_pdf(z)

    if func == AcquisitionFunc.PI:
        return _norm_cdf(z)

    return mu  # fallback


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class TrialResult:
    """Result of a single evaluation trial."""
    trial_id: int
    params: Dict[str, Any]
    objectives: Dict[str, float]   # e.g. {"sharpe": 1.5, "max_dd": 0.12}
    primary_value: float           # main objective value
    is_best: bool = False
    source: str = "optimizer"      # "optimizer" or "warm_start"


@dataclass
class ConvergencePoint:
    """One point on the convergence curve."""
    trial: int
    best_so_far: float
    current_value: float


@dataclass
class ParamSensitivity:
    """Sensitivity of objective to one parameter."""
    param_name: str
    correlation: float        # Spearman correlation with objective
    importance: float         # normalised importance (0-1)
    best_value: Any


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    params: Dict[str, Any]
    objectives: Dict[str, float]
    is_dominated: bool = False


@dataclass
class OptimizerResult:
    """Complete optimization output."""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_value: float = 0.0
    best_objectives: Dict[str, float] = field(default_factory=dict)
    trials: List[TrialResult] = field(default_factory=list)
    convergence: List[ConvergencePoint] = field(default_factory=list)
    sensitivities: List[ParamSensitivity] = field(default_factory=list)
    pareto_frontier: List[ParetoPoint] = field(default_factory=list)
    n_trials: int = 0
    generated_at: str = ""


# ── Core optimizer ──────────────────────────────────────────────────────────
class ConfigOptimizer:
    """Bayesian optimizer for experiment configuration parameters."""

    def __init__(
        self,
        param_space: List[ParamDef],
        acquisition: str = AcquisitionFunc.UCB,
        kappa: float = 2.0,
        xi: float = 0.01,
        n_initial: int = 5,
        random_state: int = 42,
    ) -> None:
        self.param_space = param_space
        self.acquisition = acquisition
        self.kappa = kappa
        self.xi = xi
        self.n_initial = n_initial
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._trials: List[TrialResult] = []
        self._gp: Optional[GaussianProcessRegressor] = None

    # ── Public API ──────────────────────────────────────────────────────────
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 30,
        primary_objective: str = "sharpe",
        warm_start: Optional[List[Dict[str, Any]]] = None,
    ) -> OptimizerResult:
        """Run Bayesian optimization loop.

        Parameters
        ----------
        objective_fn : callable
            Takes param dict, returns dict of objectives (e.g. {"sharpe": 1.5, "max_dd": 0.12}).
        n_trials : int
            Total trials including initial random and warm-start.
        primary_objective : str
            Key in objectives dict to maximise.
        warm_start : list of dict, optional
            Previous results: [{"params": {...}, "objectives": {...}}, ...].
        """
        # Warm start
        if warm_start:
            for ws in warm_start:
                params = ws["params"]
                objectives = ws["objectives"]
                pv = objectives.get(primary_objective, 0.0)
                self._record(params, objectives, pv, source="warm_start")

        total_done = len(self._trials)
        remaining = max(0, n_trials - total_done)

        # Initial random exploration
        n_random = max(0, self.n_initial - total_done)
        n_random = min(n_random, remaining)

        for _ in range(n_random):
            params = self._random_sample()
            objectives = objective_fn(params)
            pv = objectives.get(primary_objective, 0.0)
            self._record(params, objectives, pv)
        remaining -= n_random

        # Bayesian iterations
        for _ in range(remaining):
            self._fit_gp()
            params = self._suggest()
            objectives = objective_fn(params)
            pv = objectives.get(primary_objective, 0.0)
            self._record(params, objectives, pv)

        # Post-processing
        best_trial = max(self._trials, key=lambda t: t.primary_value)
        best_trial.is_best = True

        convergence = self._convergence_curve()
        sensitivities = self._param_sensitivity(primary_objective)
        pareto = self._pareto_frontier()

        return OptimizerResult(
            best_params=best_trial.params,
            best_value=best_trial.primary_value,
            best_objectives=best_trial.objectives,
            trials=list(self._trials),
            convergence=convergence,
            sensitivities=sensitivities,
            pareto_frontier=pareto,
            n_trials=len(self._trials),
            generated_at=self._now(),
        )

    def suggest(self) -> Dict[str, Any]:
        """Suggest next parameters to evaluate (for manual loop)."""
        if len(self._X) < self.n_initial:
            return self._random_sample()
        self._fit_gp()
        return self._suggest()

    def tell(
        self,
        params: Dict[str, Any],
        objectives: Dict[str, float],
        primary_objective: str = "sharpe",
    ) -> None:
        """Record an externally-evaluated result."""
        pv = objectives.get(primary_objective, 0.0)
        self._record(params, objectives, pv)

    def generate_report(
        self,
        result: OptimizerResult,
        output_path: str | Path = "reports/config_optimizer.html",
    ) -> Path:
        """Write HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Optimizer report written to %s", path)
        return path

    # ── Internals ───────────────────────────────────────────────────────────
    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.array([p.encode(params.get(p.name, p.default)) for p in self.param_space])

    def _decode_vector(self, vec: np.ndarray) -> Dict[str, Any]:
        return {p.name: p.decode(float(vec[i])) for i, p in enumerate(self.param_space)}

    def _random_sample(self) -> Dict[str, Any]:
        return {p.name: p.sample(self._rng) for p in self.param_space}

    def _record(
        self, params: Dict[str, Any], objectives: Dict[str, float],
        primary_value: float, source: str = "optimizer",
    ) -> None:
        x = self._encode_params(params)
        self._X.append(x)
        self._y.append(primary_value)
        self._trials.append(TrialResult(
            trial_id=len(self._trials),
            params=dict(params),
            objectives=dict(objectives),
            primary_value=primary_value,
            source=source,
        ))

    def _fit_gp(self) -> None:
        if len(self._X) < 2:
            return
        X = np.vstack(self._X)
        y = np.array(self._y)
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        self._gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2,
            random_state=self.random_state, normalize_y=True,
        )
        self._gp.fit(X, y)

    def _suggest(self) -> Dict[str, Any]:
        if self._gp is None:
            return self._random_sample()

        # Generate candidates
        n_candidates = 500
        candidates = np.array([
            self._encode_params(self._random_sample())
            for _ in range(n_candidates)
        ])

        mu, sigma = self._gp.predict(candidates, return_std=True)
        best_y = max(self._y) if self._y else 0.0

        acq = compute_acquisition(mu, sigma, best_y, self.acquisition, self.kappa, self.xi)
        best_idx = int(np.argmax(acq))
        return self._decode_vector(candidates[best_idx])

    # ── Analysis ────────────────────────────────────────────────────────────
    def _convergence_curve(self) -> List[ConvergencePoint]:
        points: List[ConvergencePoint] = []
        best = -np.inf
        for i, t in enumerate(self._trials):
            best = max(best, t.primary_value)
            points.append(ConvergencePoint(
                trial=i, best_so_far=best, current_value=t.primary_value,
            ))
        return points

    def _param_sensitivity(self, primary: str) -> List[ParamSensitivity]:
        if len(self._trials) < 5:
            return []
        results: List[ParamSensitivity] = []
        y = np.array([t.primary_value for t in self._trials])

        for p in self.param_space:
            vals = np.array([p.encode(t.params.get(p.name, p.default)) for t in self._trials])
            if np.std(vals) < 1e-12:
                results.append(ParamSensitivity(p.name, 0.0, 0.0, p.default))
                continue
            # Spearman rank correlation
            rank_x = _rankdata(vals)
            rank_y = _rankdata(y)
            n = len(vals)
            d_sq = np.sum((rank_x - rank_y) ** 2)
            corr = 1 - 6 * d_sq / (n * (n * n - 1)) if n > 1 else 0.0

            best_idx = int(np.argmax(y))
            best_val = self._trials[best_idx].params.get(p.name, p.default)
            results.append(ParamSensitivity(p.name, float(corr), abs(float(corr)), best_val))

        # Normalise importance
        total = sum(s.importance for s in results) or 1.0
        for s in results:
            s.importance /= total

        return sorted(results, key=lambda s: -s.importance)

    def _pareto_frontier(self) -> List[ParetoPoint]:
        if len(self._trials) < 2:
            return []
        # Collect all objective keys from first trial
        obj_keys = list(self._trials[0].objectives.keys())
        if len(obj_keys) < 2:
            return []

        points: List[ParetoPoint] = []
        for t in self._trials:
            points.append(ParetoPoint(params=t.params, objectives=t.objectives))

        # Mark dominated points (assume maximise all objectives)
        n = len(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                obj_i = points[i].objectives
                obj_j = points[j].objectives
                # j dominates i if j >= i on all and j > i on at least one
                all_geq = all(obj_j.get(k, 0) >= obj_i.get(k, 0) for k in obj_keys)
                any_gt = any(obj_j.get(k, 0) > obj_i.get(k, 0) for k in obj_keys)
                if all_geq and any_gt:
                    points[i].is_dominated = True
                    break

        return points

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML ────────────────────────────────────────────────────────────────
    def _build_html(self, r: OptimizerResult) -> str:
        cards = self._html_cards(r)
        conv = self._svg_convergence(r.convergence)
        sens = self._svg_sensitivity(r.sensitivities)
        pareto = self._svg_pareto(r.pareto_frontier)
        best_tbl = self._html_best_params(r)
        trials_tbl = self._html_trials(r.trials)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Config Optimizer</title>
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
.pos{{color:#4ade80}}.neg{{color:#f87171}}
svg{{display:block;margin:0 auto}}
</style>
</head>
<body>
<h1>Bayesian Config Optimizer</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_trials} trials</p>

{cards}
{best_tbl}

<div class="sec"><h2>Convergence</h2>{conv}</div>
<div class="sec"><h2>Parameter Sensitivity</h2>{sens}</div>
<div class="sec"><h2>Pareto Frontier</h2>{pareto}</div>

{trials_tbl}
</body>
</html>"""

    @staticmethod
    def _html_cards(r: OptimizerResult) -> str:
        n_pareto = sum(1 for p in r.pareto_frontier if not p.is_dominated)
        return f"""<div class="grid">
<div class="card"><div class="lbl">Best Value</div><div class="val pos">{r.best_value:.4f}</div></div>
<div class="card"><div class="lbl">Trials</div><div class="val">{r.n_trials}</div></div>
<div class="card"><div class="lbl">Pareto Front</div><div class="val">{n_pareto}</div></div>
</div>"""

    @staticmethod
    def _html_best_params(r: OptimizerResult) -> str:
        if not r.best_params:
            return ""
        rows = "".join(f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in sorted(r.best_params.items()))
        obj_rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(r.best_objectives.items()))
        return f"""<div class="sec">
<h2>Best Configuration</h2>
<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>
<h2 style="margin-top:16px">Objectives</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{obj_rows}</tbody></table>
</div>"""

    @staticmethod
    def _svg_convergence(conv: List[ConvergencePoint]) -> str:
        if not conv:
            return "<p>No data.</p>"
        w, h = 520, 200
        pl, pb, pt = 50, 30, 15
        cw, ch = w - pl, h - pb - pt
        n = len(conv)
        ys = [c.best_so_far for c in conv]
        min_y, max_y = min(ys), max(ys)
        rng_y = max_y - min_y or 1.0

        pts = []
        for i, c in enumerate(conv):
            x = pl + i / max(n - 1, 1) * cw
            y = pt + ch - (c.best_so_far - min_y) / rng_y * ch
            pts.append(f"{x:.0f},{y:.0f}")

        dots = "".join(
            f'<circle cx="{pl + i / max(n - 1, 1) * cw:.0f}" cy="{pt + ch - (c.current_value - min_y) / rng_y * ch:.0f}" r="3" fill="#94a3b8" opacity="0.5"/>'
            for i, c in enumerate(conv)
        )

        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pl}" y1="{pt + ch}" x2="{w}" y2="{pt + ch}" stroke="#475569" stroke-width="1"/>'
            f'{dots}'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4ade80" stroke-width="2"/>'
            f'<text x="{pl - 5}" y="{pt + 4}" text-anchor="end" font-size="10" fill="#94a3b8">{max_y:.3f}</text>'
            f'<text x="{pl - 5}" y="{pt + ch}" text-anchor="end" font-size="10" fill="#94a3b8">{min_y:.3f}</text>'
            f'</svg>'
        )

    @staticmethod
    def _svg_sensitivity(sens: List[ParamSensitivity]) -> str:
        if not sens:
            return "<p>No data.</p>"
        w = 480
        h_chart = 28 * len(sens) + 20
        pl = 140
        max_imp = max(s.importance for s in sens) or 1.0
        bars = ""
        for i, s in enumerate(sens):
            y = 10 + i * 28
            bw = max(2, s.importance / max_imp * (w - pl - 40))
            colour = "#4ade80" if s.correlation >= 0 else "#f87171"
            bars += (
                f'<text x="{pl - 5}" y="{y + 14}" text-anchor="end" font-size="11" fill="#e2e8f0">{s.param_name}</text>'
                f'<rect x="{pl}" y="{y}" width="{bw:.0f}" height="20" rx="3" fill="{colour}" opacity="0.8"/>'
                f'<text x="{pl + bw + 5}" y="{y + 14}" font-size="10" fill="#94a3b8">{s.importance:.2f} (r={s.correlation:+.2f})</text>'
            )
        return f'<svg viewBox="0 0 {w} {h_chart}" width="{w}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

    @staticmethod
    def _svg_pareto(pareto: List[ParetoPoint]) -> str:
        if not pareto:
            return "<p>No multi-objective data.</p>"
        obj_keys = list(pareto[0].objectives.keys())
        if len(obj_keys) < 2:
            return "<p>Need 2+ objectives for Pareto.</p>"
        k1, k2 = obj_keys[0], obj_keys[1]

        w, h = 350, 350
        pad = 45
        ch = h - 2 * pad
        xs = [p.objectives.get(k1, 0) for p in pareto]
        ys_raw = [p.objectives.get(k2, 0) for p in pareto]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys_raw), max(ys_raw)
        rx = max_x - min_x or 1
        ry = max_y - min_y or 1

        dots = ""
        for p in pareto:
            px = pad + (p.objectives.get(k1, 0) - min_x) / rx * ch
            py = h - pad - (p.objectives.get(k2, 0) - min_y) / ry * ch
            colour = "#4ade80" if not p.is_dominated else "#475569"
            r = 5 if not p.is_dominated else 3
            dots += f'<circle cx="{px:.0f}" cy="{py:.0f}" r="{r}" fill="{colour}"/>'

        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#475569" stroke-width="1"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#475569" stroke-width="1"/>'
            f'{dots}'
            f'<text x="{w // 2}" y="{h - 5}" text-anchor="middle" font-size="10" fill="#94a3b8">{k1}</text>'
            f'<text x="10" y="{h // 2}" font-size="10" fill="#94a3b8" transform="rotate(-90 10 {h // 2})">{k2}</text>'
            f'</svg>'
        )

    @staticmethod
    def _html_trials(trials: List[TrialResult]) -> str:
        if not trials:
            return ""
        rows = ""
        for t in sorted(trials, key=lambda t: -t.primary_value)[:20]:
            obj_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(t.objectives.items()))
            src = f"({'warm_start' if t.source == 'warm_start' else t.trial_id})"
            cls = "pos" if t.is_best else ""
            rows += f'<tr class="{cls}"><td>{t.trial_id}</td><td>{t.primary_value:.4f}</td><td>{obj_str}</td><td>{src}</td></tr>'
        return f"""<div class="sec">
<h2>Top Trials</h2>
<table><thead><tr><th>#</th><th>Primary</th><th>Objectives</th><th>Source</th></tr></thead><tbody>{rows}</tbody></table>
</div>"""


# ── Utility ─────────────────────────────────────────────────────────────────
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Simple rank (average ties)."""
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks
