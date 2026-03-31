"""
Multi-criteria experiment ranking system.

Metrics:
  Sharpe, annual return, max drawdown, Calmar, Sortino, hit rate,
  profit factor, capacity score, signal decay rate, OOS degradation.

Features:
  - Composite scoring with configurable weights
  - S/A/B/C/D/F tier classification
  - Promotion criteria (paper → live)
  - Demotion triggers (live → paper / retire)
  - Historical rank tracking across snapshots
  - HTML leaderboard with radar charts per experiment

Usage::

    from compass.experiment_ranker import ExperimentRanker
    ranker = ExperimentRanker()
    result = ranker.rank(experiments)
    ExperimentRanker.generate_report(result)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "experiment_ranker.html"
TRADING_DAYS = 252

TIERS = ["S", "A", "B", "C", "D", "F"]

DEFAULT_WEIGHTS = {
    "sharpe": 0.20,
    "annual_return": 0.15,
    "max_drawdown": 0.12,
    "calmar": 0.08,
    "sortino": 0.10,
    "hit_rate": 0.10,
    "profit_factor": 0.08,
    "capacity": 0.05,
    "signal_decay": 0.06,
    "oos_degradation": 0.06,
}

TIER_THRESHOLDS = {
    "S": 0.85,
    "A": 0.70,
    "B": 0.55,
    "C": 0.40,
    "D": 0.25,
}

PROMOTION_CRITERIA = {
    "min_tier": "B",
    "min_sharpe": 1.5,
    "min_hit_rate": 0.50,
    "max_drawdown_pct": 20.0,
    "min_trades": 30,
    "max_oos_degradation": 0.40,
}

DEMOTION_TRIGGERS = {
    "min_tier": "D",
    "max_drawdown_pct": 30.0,
    "min_sharpe": 0.0,
    "min_hit_rate": 0.35,
}


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ExperimentMetrics:
    """Raw metrics for one experiment."""

    name: str
    sharpe: float = 0.0
    annual_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar: float = 0.0
    sortino: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    capacity_score: float = 0.5
    signal_decay_rate: float = 0.0
    oos_degradation: float = 0.0
    n_trades: int = 0
    total_pnl: float = 0.0


@dataclass
class NormalizedScores:
    """Percentile-normalized scores (0-1) for one experiment."""

    name: str
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    sortino: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    capacity: float = 0.0
    signal_decay: float = 0.0
    oos_degradation: float = 0.0
    composite: float = 0.0


@dataclass
class TierClassification:
    """Tier assignment for one experiment."""

    name: str
    tier: str
    composite_score: float
    rank: int
    promote_eligible: bool
    demote_triggered: bool
    promote_reasons: List[str]
    demote_reasons: List[str]


@dataclass
class RankSnapshot:
    """Historical ranking at a point in time."""

    timestamp: str
    rankings: List[TierClassification]


@dataclass
class RankerResult:
    """Full ranking result."""

    experiments: List[ExperimentMetrics]
    normalized: List[NormalizedScores]
    classifications: List[TierClassification]
    weights: Dict[str, float]
    history: List[RankSnapshot]
    n_experiments: int
    tier_counts: Dict[str, int]


# ── Metric computation from trade data ───────────────────────────────────


def compute_metrics_from_trades(
    name: str,
    trades: pd.DataFrame,
    capital: float = 100_000.0,
) -> ExperimentMetrics:
    """Compute all ranking metrics from trade data."""
    if trades.empty or "pnl" not in trades.columns:
        return ExperimentMetrics(name=name)

    pnls = trades["pnl"].values.astype(float)
    n = len(pnls)
    if n == 0:
        return ExperimentMetrics(name=name)

    # Basic
    total_pnl = float(pnls.sum())
    wins = (pnls > 0).sum()
    hit_rate = float(wins / n)
    gains = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    pf = float(gains / losses) if losses > 1e-12 else (10.0 if gains > 0 else 0.0)
    pf = min(pf, 10.0)

    # Annualized
    annual_ret = total_pnl / capital * 100

    # Sharpe
    mu = pnls.mean()
    std = pnls.std(ddof=1) if n > 1 else 1.0
    sharpe = float(mu / std * math.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0

    # Sortino
    down = pnls[pnls < 0]
    ds = np.sqrt(np.mean(down ** 2)) if len(down) > 0 else 1.0
    sortino = float(mu / ds * math.sqrt(TRADING_DAYS)) if ds > 1e-12 else 0.0

    # Max drawdown
    equity = capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(abs(dd.min()) * 100)

    # Calmar
    calmar = float(annual_ret / max_dd) if max_dd > 1e-6 else 0.0

    # Capacity: proxy from avg trade size stability
    if n > 10:
        half = n // 2
        first_half_pnl = pnls[:half].mean()
        second_half_pnl = pnls[half:].mean()
        if abs(first_half_pnl) > 1e-12:
            capacity = min(1.0, max(0.0, second_half_pnl / first_half_pnl))
        else:
            capacity = 0.5
    else:
        capacity = 0.5

    # Signal decay: trend of rolling win rate
    decay = 0.0
    if n > 20:
        rolling_wr = pd.Series(pnls > 0).rolling(min(20, n // 3)).mean().dropna()
        if len(rolling_wr) > 5:
            x = np.arange(len(rolling_wr), dtype=float)
            slope = float(np.polyfit(x, rolling_wr.values, 1)[0])
            decay = max(0.0, -slope * 100)  # positive = decaying

    # OOS degradation: second half vs first half Sharpe
    oos_deg = 0.0
    if n > 20:
        half = n // 2
        pnls_1, pnls_2 = pnls[:half], pnls[half:]
        s1 = pnls_1.mean() / (pnls_1.std(ddof=1) + 1e-12) * math.sqrt(TRADING_DAYS)
        s2 = pnls_2.mean() / (pnls_2.std(ddof=1) + 1e-12) * math.sqrt(TRADING_DAYS)
        if abs(s1) > 1e-6:
            oos_deg = max(0.0, 1.0 - s2 / s1)
        oos_deg = min(oos_deg, 1.0)

    return ExperimentMetrics(
        name=name, sharpe=sharpe, annual_return_pct=annual_ret,
        max_drawdown_pct=max_dd, calmar=calmar, sortino=sortino,
        hit_rate=hit_rate, profit_factor=pf, capacity_score=capacity,
        signal_decay_rate=decay, oos_degradation=oos_deg,
        n_trades=n, total_pnl=total_pnl,
    )


# ── Normalization ────────────────────────────────────────────────────────


def normalize_metrics(
    experiments: List[ExperimentMetrics],
) -> List[NormalizedScores]:
    """Normalize each metric to 0-1 percentile across experiments."""
    if not experiments:
        return []

    def _rank_normalize(values: List[float], higher_better: bool = True) -> List[float]:
        n = len(values)
        if n <= 1:
            return [0.5] * n
        arr = np.array(values)
        ranks = np.argsort(np.argsort(arr)).astype(float)
        normalized = ranks / (n - 1) if n > 1 else np.full(n, 0.5)
        if not higher_better:
            normalized = 1.0 - normalized
        return normalized.tolist()

    sharpes = _rank_normalize([e.sharpe for e in experiments])
    returns = _rank_normalize([e.annual_return_pct for e in experiments])
    drawdowns = _rank_normalize([e.max_drawdown_pct for e in experiments], higher_better=False)
    calmars = _rank_normalize([e.calmar for e in experiments])
    sortinos = _rank_normalize([e.sortino for e in experiments])
    hit_rates = _rank_normalize([e.hit_rate for e in experiments])
    pfs = _rank_normalize([e.profit_factor for e in experiments])
    caps = _rank_normalize([e.capacity_score for e in experiments])
    decays = _rank_normalize([e.signal_decay_rate for e in experiments], higher_better=False)
    oos = _rank_normalize([e.oos_degradation for e in experiments], higher_better=False)

    results: List[NormalizedScores] = []
    for i, e in enumerate(experiments):
        results.append(NormalizedScores(
            name=e.name,
            sharpe=sharpes[i], annual_return=returns[i],
            max_drawdown=drawdowns[i], calmar=calmars[i],
            sortino=sortinos[i], hit_rate=hit_rates[i],
            profit_factor=pfs[i], capacity=caps[i],
            signal_decay=decays[i], oos_degradation=oos[i],
            composite=0.0,
        ))
    return results


def compute_composite(
    normalized: List[NormalizedScores],
    weights: Dict[str, float],
) -> List[NormalizedScores]:
    """Compute weighted composite score."""
    for ns in normalized:
        score = (
            weights.get("sharpe", 0) * ns.sharpe
            + weights.get("annual_return", 0) * ns.annual_return
            + weights.get("max_drawdown", 0) * ns.max_drawdown
            + weights.get("calmar", 0) * ns.calmar
            + weights.get("sortino", 0) * ns.sortino
            + weights.get("hit_rate", 0) * ns.hit_rate
            + weights.get("profit_factor", 0) * ns.profit_factor
            + weights.get("capacity", 0) * ns.capacity
            + weights.get("signal_decay", 0) * ns.signal_decay
            + weights.get("oos_degradation", 0) * ns.oos_degradation
        )
        ns.composite = score
    return normalized


# ── Tier classification ──────────────────────────────────────────────────


def classify_tier(composite: float) -> str:
    """Classify composite score into tier."""
    for tier in ["S", "A", "B", "C", "D"]:
        if composite >= TIER_THRESHOLDS[tier]:
            return tier
    return "F"


def check_promotion(
    metrics: ExperimentMetrics,
    tier: str,
    criteria: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Check if experiment meets promotion criteria."""
    c = criteria or PROMOTION_CRITERIA
    reasons: List[str] = []
    eligible = True

    min_tier_idx = TIERS.index(c.get("min_tier", "B"))
    if TIERS.index(tier) > min_tier_idx:
        eligible = False
        reasons.append(f"tier {tier} below minimum {c['min_tier']}")

    if metrics.sharpe < c.get("min_sharpe", 1.5):
        eligible = False
        reasons.append(f"Sharpe {metrics.sharpe:.2f} < {c['min_sharpe']}")

    if metrics.hit_rate < c.get("min_hit_rate", 0.50):
        eligible = False
        reasons.append(f"hit rate {metrics.hit_rate:.1%} < {c['min_hit_rate']:.1%}")

    if metrics.max_drawdown_pct > c.get("max_drawdown_pct", 20.0):
        eligible = False
        reasons.append(f"max DD {metrics.max_drawdown_pct:.1f}% > {c['max_drawdown_pct']}%")

    if metrics.n_trades < c.get("min_trades", 30):
        eligible = False
        reasons.append(f"only {metrics.n_trades} trades (min {c['min_trades']})")

    if metrics.oos_degradation > c.get("max_oos_degradation", 0.40):
        eligible = False
        reasons.append(f"OOS degradation {metrics.oos_degradation:.1%} > {c['max_oos_degradation']:.1%}")

    return eligible, reasons


def check_demotion(
    metrics: ExperimentMetrics,
    tier: str,
    triggers: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Check if experiment triggers demotion."""
    t = triggers or DEMOTION_TRIGGERS
    reasons: List[str] = []
    triggered = False

    min_tier_idx = TIERS.index(t.get("min_tier", "D"))
    if TIERS.index(tier) >= min_tier_idx:
        triggered = True
        reasons.append(f"tier {tier} at or below {t['min_tier']}")

    if metrics.max_drawdown_pct > t.get("max_drawdown_pct", 30.0):
        triggered = True
        reasons.append(f"max DD {metrics.max_drawdown_pct:.1f}% exceeds {t['max_drawdown_pct']}%")

    if metrics.sharpe < t.get("min_sharpe", 0.0):
        triggered = True
        reasons.append(f"Sharpe {metrics.sharpe:.2f} below {t['min_sharpe']}")

    if metrics.hit_rate < t.get("min_hit_rate", 0.35):
        triggered = True
        reasons.append(f"hit rate {metrics.hit_rate:.1%} below {t['min_hit_rate']:.1%}")

    return triggered, reasons


# ── Core ranker ──────────────────────────────────────────────────────────


class ExperimentRanker:
    """Multi-criteria experiment ranking system."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        promotion_criteria: Optional[Dict[str, Any]] = None,
        demotion_triggers: Optional[Dict[str, Any]] = None,
    ):
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.promotion_criteria = promotion_criteria or dict(PROMOTION_CRITERIA)
        self.demotion_triggers = demotion_triggers or dict(DEMOTION_TRIGGERS)
        self._history: List[RankSnapshot] = []

    def rank(
        self,
        experiments: List[ExperimentMetrics],
    ) -> RankerResult:
        """Rank experiments and classify tiers."""
        if not experiments:
            return RankerResult([], [], [], self.weights, self._history, 0, {})

        # Normalize
        normalized = normalize_metrics(experiments)
        normalized = compute_composite(normalized, self.weights)

        # Sort by composite descending
        order = sorted(range(len(normalized)), key=lambda i: normalized[i].composite, reverse=True)
        sorted_norm = [normalized[i] for i in order]
        sorted_exp = [experiments[i] for i in order]

        # Classify
        classifications: List[TierClassification] = []
        for rank_idx, (ns, em) in enumerate(zip(sorted_norm, sorted_exp)):
            tier = classify_tier(ns.composite)
            promo_ok, promo_reasons = check_promotion(em, tier, self.promotion_criteria)
            demo_ok, demo_reasons = check_demotion(em, tier, self.demotion_triggers)

            classifications.append(TierClassification(
                name=ns.name, tier=tier, composite_score=ns.composite,
                rank=rank_idx + 1,
                promote_eligible=promo_ok, demote_triggered=demo_ok,
                promote_reasons=[] if promo_ok else promo_reasons,
                demote_reasons=demo_reasons if demo_ok else [],
            ))

        # Tier counts
        tier_counts = {t: 0 for t in TIERS}
        for c in classifications:
            tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1

        # Record history
        snapshot = RankSnapshot(
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            rankings=list(classifications),
        )
        self._history.append(snapshot)

        return RankerResult(
            experiments=sorted_exp,
            normalized=sorted_norm,
            classifications=classifications,
            weights=self.weights,
            history=list(self._history),
            n_experiments=len(experiments),
            tier_counts=tier_counts,
        )

    def rank_from_trades(
        self,
        experiment_trades: Dict[str, pd.DataFrame],
        capital: float = 100_000.0,
    ) -> RankerResult:
        """Convenience: compute metrics from trade data then rank."""
        experiments = [
            compute_metrics_from_trades(name, df, capital)
            for name, df in experiment_trades.items()
        ]
        return self.rank(experiments)

    @staticmethod
    def generate_report(
        result: RankerResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fr(v: float) -> str:
    return f"{v:.2f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


TIER_COLORS = {
    "S": "#ff6b6b", "A": "#ffa502", "B": "#2ed573",
    "C": "#58a6ff", "D": "#8b949e", "F": "#f85149",
}


def _radar_svg(ns: NormalizedScores, w: int = 200, h: int = 200) -> str:
    """Small radar chart for one experiment."""
    metrics = [
        ("Sharpe", ns.sharpe), ("Return", ns.annual_return),
        ("DD", ns.max_drawdown), ("Sortino", ns.sortino),
        ("HitRate", ns.hit_rate), ("PF", ns.profit_factor),
        ("Capacity", ns.capacity), ("Decay", ns.signal_decay),
    ]
    n = len(metrics)
    cx, cy, r = w // 2, h // 2, min(w, h) // 2 - 25
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]

    parts = [f'<svg viewBox="0 0 {w} {h}" class="radar">']
    # Grid rings
    for ring in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{cx + r * ring * math.cos(a):.0f},{cy + r * ring * math.sin(a):.0f}" for a in angles)
        parts.append(f'<polygon points="{pts}" fill="none" stroke="#21262d" stroke-width="0.5"/>')

    # Data polygon
    data_pts = " ".join(
        f"{cx + r * v * math.cos(angles[i]):.0f},{cy + r * v * math.sin(angles[i]):.0f}"
        for i, (_, v) in enumerate(metrics)
    )
    parts.append(f'<polygon points="{data_pts}" fill="#58a6ff" fill-opacity="0.2" stroke="#58a6ff" stroke-width="1.5"/>')

    # Labels
    for i, (label, _) in enumerate(metrics):
        lx = cx + (r + 15) * math.cos(angles[i])
        ly = cy + (r + 15) * math.sin(angles[i])
        parts.append(f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" font-size="7" fill="#8b949e">{label}</text>')

    # Title
    parts.append(f'<text x="{cx}" y="{h - 3}" text-anchor="middle" font-size="9" fill="#c9d1d9">{ns.name} ({_fr(ns.composite)})</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _leaderboard_table(result: RankerResult) -> str:
    rows = ""
    for cls, em, ns in zip(result.classifications, result.experiments, result.normalized):
        tc = TIER_COLORS.get(cls.tier, "#8b949e")
        promo = "✓" if cls.promote_eligible else ""
        demo = "⚠" if cls.demote_triggered else ""
        rows += f"""<tr>
          <td>{cls.rank}</td>
          <td style="text-align:left"><span style="color:{tc};font-weight:700">[{cls.tier}]</span> {cls.name}</td>
          <td>{_fr(cls.composite_score)}</td>
          <td>{_fr(em.sharpe)}</td>
          <td>{_fr(em.annual_return_pct)}%</td>
          <td>{_fr(em.max_drawdown_pct)}%</td>
          <td>{_fr(em.sortino)}</td>
          <td>{_fp(em.hit_rate)}</td>
          <td>{_fr(em.profit_factor)}</td>
          <td>{em.n_trades}</td>
          <td style="color:#3fb950">{promo}</td>
          <td style="color:#f85149">{demo}</td>
        </tr>"""
    return f"""<table class="dt">
      <tr><th>#</th><th style="text-align:left">Experiment</th><th>Score</th>
          <th>Sharpe</th><th>Return</th><th>Max DD</th><th>Sortino</th>
          <th>Hit Rate</th><th>PF</th><th>Trades</th><th>Promo</th><th>Demo</th></tr>
      {rows}</table>"""


def _build_html(result: RankerResult) -> str:
    # Tier summary
    tier_badges = " ".join(
        f'<span style="background:{TIER_COLORS.get(t, "#8b949e")};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-weight:700;margin:0 4px">{t}: {result.tier_counts.get(t, 0)}</span>'
        for t in TIERS if result.tier_counts.get(t, 0) > 0
    )

    # Radar charts (top 6)
    radars = ""
    for ns in result.normalized[:6]:
        radars += _radar_svg(ns)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><title>Experiment Leaderboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
  h1,h2 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .tiers {{ margin: 20px 0; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.dt th, table.dt td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; }}
  .radars {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; margin: 16px 0; }}
  .radar {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; }}
</style>
</head>
<body>
<h1>Experiment Leaderboard</h1>
<p class="meta">{result.n_experiments} experiments ranked &middot;
   {len(result.history)} historical snapshots</p>

<div class="tiers">{tier_badges}</div>

<h2>Leaderboard</h2>
{_leaderboard_table(result)}

<h2>Radar Charts (Top 6)</h2>
<div class="radars">{radars}</div>

</body></html>"""
