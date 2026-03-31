"""Signal decay analyzer – measures ML signal quality degradation over time.

Computes information coefficient (IC) at various holding periods, signal-to-noise
ratio, optimal holding period, turnover/flip analysis, and half-life estimation
via exponential fit.  Supports per-regime breakdown and self-contained HTML reporting.
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
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# ── Holding periods in hours ────────────────────────────────────────────────
DEFAULT_HOLDING_PERIODS: Dict[str, int] = {
    "1h": 1,
    "4h": 4,
    "1d": 24,
    "2d": 48,
    "5d": 120,
}


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class ICResult:
    """Information coefficient at a single holding period."""
    period_label: str
    period_hours: int
    ic: float
    ic_std: float
    ic_ir: float  # IC information ratio = mean(IC) / std(IC)
    n_obs: int


@dataclass
class TurnoverResult:
    """Turnover / flip analysis for a signal series."""
    flip_rate: float          # fraction of periods where signal sign flips
    avg_holding_bars: float   # average bars between flips
    estimated_cost_bps: float # round-trip cost estimate in basis points


@dataclass
class HalfLifeResult:
    """Half-life of IC decay estimated via exponential fit."""
    half_life_hours: float
    decay_rate: float   # lambda in exp(-lambda * t)
    r_squared: float    # goodness of fit


@dataclass
class RegimeDecayResult:
    """Per-regime signal decay summary."""
    regime: str
    ic_by_period: Dict[str, float]
    snr: float
    optimal_period: str
    half_life_hours: float
    flip_rate: float
    n_obs: int


@dataclass
class DecayAnalysis:
    """Full signal decay analysis result."""
    ic_results: List[ICResult] = field(default_factory=list)
    snr: float = 0.0
    optimal_period: str = ""
    optimal_ic: float = 0.0
    turnover: Optional[TurnoverResult] = None
    half_life: Optional[HalfLifeResult] = None
    regime_results: List[RegimeDecayResult] = field(default_factory=list)
    generated_at: str = ""


# ── Core analyser ───────────────────────────────────────────────────────────
class SignalDecayAnalyzer:
    """Measures how quickly an ML signal's predictive power decays."""

    def __init__(
        self,
        holding_periods: Optional[Dict[str, int]] = None,
        cost_per_flip_bps: float = 2.0,
    ) -> None:
        self.holding_periods = holding_periods or dict(DEFAULT_HOLDING_PERIODS)
        self.cost_per_flip_bps = cost_per_flip_bps

    # ── public API ──────────────────────────────────────────────────────────
    def analyze(
        self,
        signals: pd.Series,
        returns: pd.Series,
        regimes: Optional[pd.Series] = None,
    ) -> DecayAnalysis:
        """Run full decay analysis.

        Parameters
        ----------
        signals : pd.Series
            Model signal/score aligned to *returns* index.
        returns : pd.Series
            Per-bar forward returns (e.g. hourly).
        regimes : pd.Series, optional
            Regime labels aligned to same index.
        """
        signals, returns = self._align(signals, returns)
        if len(signals) < 10:
            logger.warning("Too few observations (%d) for decay analysis", len(signals))
            return DecayAnalysis(generated_at=self._now())

        ic_results = self._compute_ic_curve(signals, returns)
        snr = self._compute_snr(signals)
        optimal = self._pick_optimal(ic_results)
        turnover = self._compute_turnover(signals)
        half_life = self._estimate_half_life(ic_results)

        regime_results: List[RegimeDecayResult] = []
        if regimes is not None:
            regimes = regimes.reindex(signals.index)
            regime_results = self._per_regime(signals, returns, regimes)

        return DecayAnalysis(
            ic_results=ic_results,
            snr=snr,
            optimal_period=optimal[0],
            optimal_ic=optimal[1],
            turnover=turnover,
            half_life=half_life,
            regime_results=regime_results,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        analysis: DecayAnalysis,
        output_path: str | Path = "reports/signal_decay.html",
    ) -> Path:
        """Write self-contained HTML report and return its path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(analysis)
        path.write_text(html, encoding="utf-8")
        logger.info("Signal decay report written to %s", path)
        return path

    # ── IC computation ──────────────────────────────────────────────────────
    def _compute_ic_curve(
        self, signals: pd.Series, returns: pd.Series,
    ) -> List[ICResult]:
        results: List[ICResult] = []
        for label, hours in self.holding_periods.items():
            fwd = returns.rolling(window=hours).sum().shift(-hours)
            mask = signals.notna() & fwd.notna()
            s, f = signals[mask], fwd[mask]
            if len(s) < 5:
                results.append(ICResult(label, hours, 0.0, 0.0, 0.0, 0))
                continue
            ic = float(s.corr(f, method="spearman"))
            # rolling IC for std
            roll_ic = self._rolling_ic(s, f, window=max(20, len(s) // 10))
            ic_std = float(roll_ic.std()) if len(roll_ic) > 1 else 0.0
            ic_ir = ic / ic_std if ic_std > 1e-9 else 0.0
            results.append(ICResult(label, hours, ic, ic_std, ic_ir, len(s)))
        return results

    @staticmethod
    def _rolling_ic(
        signals: pd.Series, fwd: pd.Series, window: int,
    ) -> pd.Series:
        combined = pd.DataFrame({"s": signals.values, "f": fwd.values})
        return combined["s"].rolling(window).corr(combined["f"]).dropna()

    # ── SNR ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_snr(signals: pd.Series) -> float:
        mean = float(signals.mean())
        std = float(signals.std())
        if std < 1e-12:
            return 0.0
        return abs(mean) / std

    # ── Optimal holding period ──────────────────────────────────────────────
    @staticmethod
    def _pick_optimal(ic_results: List[ICResult]) -> Tuple[str, float]:
        if not ic_results:
            return ("", 0.0)
        best = max(ic_results, key=lambda r: abs(r.ic))
        return (best.period_label, best.ic)

    # ── Turnover ────────────────────────────────────────────────────────────
    def _compute_turnover(self, signals: pd.Series) -> TurnoverResult:
        signs = np.sign(signals.values)
        flips = np.sum(signs[1:] != signs[:-1])
        n = len(signs) - 1
        flip_rate = float(flips / n) if n > 0 else 0.0
        avg_hold = (1.0 / flip_rate) if flip_rate > 1e-9 else float(n)
        cost = flip_rate * self.cost_per_flip_bps
        return TurnoverResult(
            flip_rate=flip_rate,
            avg_holding_bars=avg_hold,
            estimated_cost_bps=cost,
        )

    # ── Half-life via exponential fit ───────────────────────────────────────
    @staticmethod
    def _estimate_half_life(ic_results: List[ICResult]) -> HalfLifeResult:
        valid = [r for r in ic_results if abs(r.ic) > 1e-9 and r.period_hours > 0]
        if len(valid) < 2:
            return HalfLifeResult(half_life_hours=float("inf"), decay_rate=0.0, r_squared=0.0)

        t = np.array([r.period_hours for r in valid], dtype=float).reshape(-1, 1)
        log_ic = np.log(np.array([abs(r.ic) for r in valid], dtype=float))

        reg = LinearRegression().fit(t, log_ic)
        decay_rate = float(-reg.coef_[0])
        r_sq = float(reg.score(t, log_ic))

        if decay_rate > 1e-12:
            hl = math.log(2) / decay_rate
        else:
            hl = float("inf")

        return HalfLifeResult(half_life_hours=hl, decay_rate=decay_rate, r_squared=max(r_sq, 0.0))

    # ── Per-regime ──────────────────────────────────────────────────────────
    def _per_regime(
        self,
        signals: pd.Series,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> List[RegimeDecayResult]:
        results: List[RegimeDecayResult] = []
        for regime in sorted(regimes.dropna().unique()):
            mask = regimes == regime
            s, r = signals[mask], returns[mask]
            if len(s) < 10:
                continue
            ic_res = self._compute_ic_curve(s, r)
            snr = self._compute_snr(s)
            opt_label, _ = self._pick_optimal(ic_res)
            hl = self._estimate_half_life(ic_res)
            to = self._compute_turnover(s)
            results.append(RegimeDecayResult(
                regime=str(regime),
                ic_by_period={r.period_label: r.ic for r in ic_res},
                snr=snr,
                optimal_period=opt_label,
                half_life_hours=hl.half_life_hours,
                flip_rate=to.flip_rate,
                n_obs=len(s),
            ))
        return results

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _align(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
        idx = a.index.intersection(b.index)
        return a.loc[idx], b.loc[idx]

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, a: DecayAnalysis) -> str:
        ic_bars = self._svg_ic_bars(a.ic_results)
        cards = self._html_cards(a)
        regime_table = self._html_regime_table(a.regime_results)
        hl_section = self._html_half_life(a)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Signal Decay Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.subtitle{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .label{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .value{{font-size:1.5rem;font-weight:700;margin-top:4px}}
.section{{margin-bottom:32px}}
.section h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}
svg{{display:block;margin:0 auto}}
</style>
</head>
<body>
<h1>Signal Decay Analysis</h1>
<p class="subtitle">Generated {a.generated_at or 'N/A'}</p>

{cards}

<div class="section">
<h2>IC Decay by Holding Period</h2>
{ic_bars}
</div>

{regime_table}

{hl_section}

</body>
</html>"""

    # ── SVG bar chart ───────────────────────────────────────────────────────
    @staticmethod
    def _svg_ic_bars(ic_results: List[ICResult]) -> str:
        if not ic_results:
            return "<p>No IC data available.</p>"
        w, h = 500, 220
        pad_l, pad_b, pad_t = 50, 40, 20
        chart_h = h - pad_b - pad_t
        n = len(ic_results)
        max_ic = max(abs(r.ic) for r in ic_results) or 0.01
        bar_w = min(50, (w - pad_l) // n - 8)

        bars = ""
        for i, r in enumerate(ic_results):
            x = pad_l + i * ((w - pad_l) // n) + 10
            bar_h = abs(r.ic) / max_ic * (chart_h * 0.8)
            y = pad_t + chart_h - bar_h if r.ic >= 0 else pad_t + chart_h
            colour = "#4ade80" if r.ic >= 0 else "#f87171"
            bars += (
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
                f'rx="3" fill="{colour}" opacity="0.85"/>'
                f'<text x="{x + bar_w // 2}" y="{y - 6}" text-anchor="middle" '
                f'font-size="11" fill="#e2e8f0">{r.ic:.3f}</text>'
                f'<text x="{x + bar_w // 2}" y="{h - 10}" text-anchor="middle" '
                f'font-size="11" fill="#94a3b8">{r.period_label}</text>'
            )

        baseline_y = pad_t + chart_h
        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pad_l}" y1="{baseline_y}" x2="{w}" y2="{baseline_y}" '
            f'stroke="#475569" stroke-width="1"/>'
            f"{bars}</svg>"
        )

    # ── Cards ───────────────────────────────────────────────────────────────
    @staticmethod
    def _html_cards(a: DecayAnalysis) -> str:
        hl_str = (
            f"{a.half_life.half_life_hours:.1f}h"
            if a.half_life and math.isfinite(a.half_life.half_life_hours)
            else "∞"
        )
        flip = f"{a.turnover.flip_rate:.1%}" if a.turnover else "N/A"
        cost = f"{a.turnover.estimated_cost_bps:.1f} bps" if a.turnover else "N/A"
        return f"""<div class="grid">
<div class="card"><div class="label">SNR</div><div class="value">{a.snr:.4f}</div></div>
<div class="card"><div class="label">Optimal Period</div><div class="value">{a.optimal_period or 'N/A'}</div></div>
<div class="card"><div class="label">Peak IC</div><div class="value">{a.optimal_ic:.4f}</div></div>
<div class="card"><div class="label">Half-Life</div><div class="value">{hl_str}</div></div>
<div class="card"><div class="label">Flip Rate</div><div class="value">{flip}</div></div>
<div class="card"><div class="label">Turnover Cost</div><div class="value">{cost}</div></div>
</div>"""

    # ── Regime table ────────────────────────────────────────────────────────
    @staticmethod
    def _html_regime_table(regimes: List[RegimeDecayResult]) -> str:
        if not regimes:
            return ""
        period_labels = list(regimes[0].ic_by_period.keys())
        ic_headers = "".join(f"<th>IC {p}</th>" for p in period_labels)
        rows = ""
        for rr in regimes:
            ic_cells = ""
            for p in period_labels:
                v = rr.ic_by_period.get(p, 0.0)
                cls = "pos" if v >= 0 else "neg"
                ic_cells += f'<td class="{cls}">{v:.4f}</td>'
            hl = f"{rr.half_life_hours:.1f}h" if math.isfinite(rr.half_life_hours) else "∞"
            rows += (
                f"<tr><td>{rr.regime}</td>{ic_cells}"
                f"<td>{rr.snr:.4f}</td><td>{rr.optimal_period}</td>"
                f"<td>{hl}</td><td>{rr.flip_rate:.1%}</td>"
                f"<td>{rr.n_obs}</td></tr>"
            )
        return f"""<div class="section">
<h2>Per-Regime Breakdown</h2>
<table>
<thead><tr><th>Regime</th>{ic_headers}<th>SNR</th><th>Optimal</th><th>Half-Life</th><th>Flip Rate</th><th>N</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── Half-life ranking ───────────────────────────────────────────────────
    @staticmethod
    def _html_half_life(a: DecayAnalysis) -> str:
        if not a.regime_results:
            return ""
        ranked = sorted(a.regime_results, key=lambda r: r.half_life_hours)
        rows = ""
        for i, rr in enumerate(ranked, 1):
            hl = f"{rr.half_life_hours:.1f}h" if math.isfinite(rr.half_life_hours) else "∞"
            rows += f"<tr><td>{i}</td><td>{rr.regime}</td><td>{hl}</td></tr>"
        return f"""<div class="section">
<h2>Half-Life Ranking</h2>
<table>
<thead><tr><th>#</th><th>Regime</th><th>Half-Life</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
