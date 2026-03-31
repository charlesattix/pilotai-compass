"""Smart execution algorithm engine — TWAP, VWAP, Implementation Shortfall,
Iceberg orders, adaptive participation, urgency-based selection, execution
benchmarking, order splitting, and dark pool routing.

Provides:
  1. TWAP: time-weighted average price scheduling
  2. VWAP: volume-weighted scheduling using historical volume profile
  3. Implementation Shortfall (Almgren-Chriss): minimise expected cost + risk
  4. Iceberg orders: visible clip size with hidden remainder
  5. Adaptive participation rate
  6. Urgency-based algorithm selection
  7. Execution benchmarking (arrival price, VWAP, close)
  8. Order splitting with minimum size constraints
  9. Dark pool routing logic
  10. HTML report with algo comparison, quality, cost savings
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
MARKET_MINUTES = 390  # 6.5 hours


class AlgoType:
    TWAP = "twap"
    VWAP = "vwap"
    IS = "implementation_shortfall"
    ICEBERG = "iceberg"


class Urgency:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class SliceOrder:
    """One child order produced by the algo."""
    slice_id: int
    quantity: int
    scheduled_time_min: int      # minutes from market open
    participation_rate: float
    is_dark: bool = False        # routed to dark pool
    is_visible: bool = True      # False = hidden iceberg portion


@dataclass
class AlgoSchedule:
    """Full execution schedule from an algorithm."""
    algo: str
    total_quantity: int
    n_slices: int
    slices: List[SliceOrder]
    duration_minutes: int
    expected_participation: float
    dark_pool_pct: float


@dataclass
class BenchmarkResult:
    """Execution quality vs benchmarks."""
    arrival_cost_bps: float      # vs arrival price
    vwap_cost_bps: float         # vs VWAP
    close_cost_bps: float        # vs close
    implementation_shortfall_bps: float
    best_benchmark: str


@dataclass
class AlgoComparison:
    """Side-by-side comparison of algorithms."""
    algo: str
    expected_cost_bps: float
    expected_risk_bps: float
    total_cost_bps: float       # cost + risk
    n_slices: int
    duration_minutes: int
    participation_rate: float


@dataclass
class ExecutionResult:
    """Complete execution algo output."""
    selected_algo: str = ""
    schedule: Optional[AlgoSchedule] = None
    benchmark: Optional[BenchmarkResult] = None
    comparisons: List[AlgoComparison] = field(default_factory=list)
    cost_savings_bps: float = 0.0    # vs market order
    generated_at: str = ""


# ── Algo configuration ──────────────────────────────────────────────────────
@dataclass
class AlgoConfig:
    """Configuration for execution algorithms."""
    # Impact model (Almgren-Chriss)
    eta: float = 0.10              # temporary impact coefficient
    gamma: float = 0.05            # permanent impact coefficient
    sigma: float = 0.02            # daily volatility
    # Participation
    max_participation: float = 0.10
    adaptive_target: float = 0.05
    # Iceberg
    clip_size: int = 10            # visible portion
    # Splitting
    min_slice_qty: int = 1
    # Dark pool
    dark_pool_threshold: int = 50  # use dark pool above this qty
    dark_pool_pct: float = 0.30   # fraction routed to dark pool
    # Volume profile (normalised, 13 half-hour buckets)
    volume_profile: Optional[List[float]] = None


DEFAULT_VOLUME_PROFILE = [
    0.14, 0.09, 0.07, 0.06, 0.06, 0.05, 0.05,
    0.05, 0.06, 0.06, 0.07, 0.09, 0.15,
]


# ── Core engine ─────────────────────────────────────────────────────────────
class ExecutionAlgoEngine:
    """Smart execution algorithm engine."""

    def __init__(self, config: Optional[AlgoConfig] = None) -> None:
        self.config = config or AlgoConfig()
        self._vol_profile = self.config.volume_profile or list(DEFAULT_VOLUME_PROFILE)

    # ── Public API ──────────────────────────────────────────────────────────
    def execute(
        self,
        total_quantity: int,
        urgency: str = Urgency.MEDIUM,
        arrival_price: float = 0.0,
        vwap_price: float = 0.0,
        close_price: float = 0.0,
        avg_fill_price: float = 0.0,
        adv: int = 5000,
        side: str = "buy",
    ) -> ExecutionResult:
        """Select algorithm, generate schedule, and benchmark.

        Parameters
        ----------
        total_quantity : int — total contracts to execute
        urgency : str — low / medium / high / critical
        arrival_price : float — price at decision time (for benchmarking)
        vwap_price : float — market VWAP (for benchmarking)
        close_price : float — market close price (for benchmarking)
        avg_fill_price : float — actual average fill (for benchmarking)
        adv : int — average daily volume in contracts
        side : str — buy or sell
        """
        if total_quantity <= 0:
            return ExecutionResult(generated_at=self._now())

        # Generate schedules for all algos
        algos = {
            AlgoType.TWAP: self._twap(total_quantity, urgency, adv),
            AlgoType.VWAP: self._vwap(total_quantity, urgency, adv),
            AlgoType.IS: self._implementation_shortfall(total_quantity, urgency, adv),
            AlgoType.ICEBERG: self._iceberg(total_quantity, adv),
        }

        # Compare
        comparisons = self._compare_algos(algos, total_quantity, adv)

        # Select
        selected = self._select_algo(urgency, total_quantity, adv)
        schedule = algos[selected]

        # Benchmark
        benchmark = None
        if avg_fill_price > 0:
            benchmark = self._benchmark(
                avg_fill_price, arrival_price, vwap_price, close_price, side,
            )

        # Cost savings vs market
        market_cost = self.config.eta * math.sqrt(total_quantity / max(adv, 1)) * 10_000
        algo_cost = next((c.total_cost_bps for c in comparisons if c.algo == selected), 0)
        savings = max(0, market_cost - algo_cost)

        return ExecutionResult(
            selected_algo=selected,
            schedule=schedule,
            benchmark=benchmark,
            comparisons=comparisons,
            cost_savings_bps=savings,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: ExecutionResult,
        output_path: str | Path = "reports/execution_algo.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Execution algo report written to %s", path)
        return path

    # ── TWAP ────────────────────────────────────────────────────────────────
    def _twap(self, qty: int, urgency: str, adv: int) -> AlgoSchedule:
        duration = self._urgency_duration(urgency)
        n_slices = max(1, duration // 30)  # one slice per 30 minutes
        slice_qty = self._split_quantity(qty, n_slices)
        slices: List[SliceOrder] = []
        for i, sq in enumerate(slice_qty):
            t = int(i * (duration / n_slices))
            dark = self._should_dark(qty, adv)
            slices.append(SliceOrder(
                slice_id=i, quantity=sq, scheduled_time_min=t,
                participation_rate=sq / max(adv / 13, 1),
                is_dark=dark and i % 3 == 0,
            ))
        dark_pct = sum(1 for s in slices if s.is_dark) / len(slices) if slices else 0
        return AlgoSchedule(
            algo=AlgoType.TWAP, total_quantity=qty, n_slices=len(slices),
            slices=slices, duration_minutes=duration,
            expected_participation=qty / max(adv, 1),
            dark_pool_pct=dark_pct,
        )

    # ── VWAP ────────────────────────────────────────────────────────────────
    def _vwap(self, qty: int, urgency: str, adv: int) -> AlgoSchedule:
        duration = self._urgency_duration(urgency)
        n_buckets = len(self._vol_profile)
        bucket_minutes = MARKET_MINUTES / n_buckets
        active_buckets = max(1, int(duration / bucket_minutes))

        profile = self._vol_profile[:active_buckets]
        total_w = sum(profile) or 1.0
        normalised = [w / total_w for w in profile]

        slices: List[SliceOrder] = []
        remaining = qty
        for i, w in enumerate(normalised):
            sq = max(self.config.min_slice_qty, int(round(qty * w)))
            sq = min(sq, remaining)
            if sq <= 0:
                continue
            remaining -= sq
            t = int(i * bucket_minutes)
            dark = self._should_dark(qty, adv)
            slices.append(SliceOrder(
                slice_id=i, quantity=sq, scheduled_time_min=t,
                participation_rate=sq / max(adv * w, 1),
                is_dark=dark and w < 0.06,  # dark in low-volume buckets
            ))
        # Allocate remainder to last slice
        if remaining > 0 and slices:
            slices[-1].quantity += remaining

        dark_pct = sum(1 for s in slices if s.is_dark) / len(slices) if slices else 0
        return AlgoSchedule(
            algo=AlgoType.VWAP, total_quantity=qty, n_slices=len(slices),
            slices=slices, duration_minutes=duration,
            expected_participation=qty / max(adv, 1),
            dark_pool_pct=dark_pct,
        )

    # ── Implementation Shortfall (Almgren-Chriss) ───────────────────────────
    def _implementation_shortfall(self, qty: int, urgency: str, adv: int) -> AlgoSchedule:
        duration = self._urgency_duration(urgency)
        participation = qty / max(adv, 1)

        # Optimal trajectory: front-load if urgency high, spread if low
        risk_aversion = {"low": 0.5, "medium": 1.0, "high": 2.0, "critical": 5.0}.get(urgency, 1.0)

        n_slices = max(1, duration // 20)  # one slice per 20 min
        # Exponential decay schedule
        tau = n_slices / max(risk_aversion, 0.1)
        raw_weights = [math.exp(-i / tau) for i in range(n_slices)]
        total_w = sum(raw_weights) or 1.0
        normalised = [w / total_w for w in raw_weights]

        slices: List[SliceOrder] = []
        remaining = qty
        for i, w in enumerate(normalised):
            sq = max(self.config.min_slice_qty, int(round(qty * w)))
            sq = min(sq, remaining)
            if sq <= 0:
                continue
            remaining -= sq
            t = int(i * (duration / n_slices))
            slices.append(SliceOrder(
                slice_id=i, quantity=sq, scheduled_time_min=t,
                participation_rate=sq / max(adv / 13, 1),
                is_dark=self._should_dark(qty, adv),
            ))
        if remaining > 0 and slices:
            slices[0].quantity += remaining

        dark_pct = sum(1 for s in slices if s.is_dark) / len(slices) if slices else 0
        return AlgoSchedule(
            algo=AlgoType.IS, total_quantity=qty, n_slices=len(slices),
            slices=slices, duration_minutes=duration,
            expected_participation=participation,
            dark_pool_pct=dark_pct,
        )

    # ── Iceberg ─────────────────────────────────────────────────────────────
    def _iceberg(self, qty: int, adv: int) -> AlgoSchedule:
        clip = self.config.clip_size
        n_clips = max(1, math.ceil(qty / clip))
        slices: List[SliceOrder] = []
        remaining = qty
        interval = MARKET_MINUTES / max(n_clips, 1)
        for i in range(n_clips):
            visible = min(clip, remaining)
            hidden = 0
            if remaining > clip:
                hidden = min(clip, remaining - clip)
            remaining -= (visible + hidden)

            t = int(i * interval)
            slices.append(SliceOrder(
                slice_id=len(slices), quantity=visible,
                scheduled_time_min=t, participation_rate=visible / max(adv / 13, 1),
                is_visible=True,
            ))
            if hidden > 0:
                slices.append(SliceOrder(
                    slice_id=len(slices), quantity=hidden,
                    scheduled_time_min=t, participation_rate=hidden / max(adv / 13, 1),
                    is_visible=False, is_dark=True,
                ))
            if remaining <= 0:
                break

        dark_pct = sum(1 for s in slices if s.is_dark) / len(slices) if slices else 0
        return AlgoSchedule(
            algo=AlgoType.ICEBERG, total_quantity=qty, n_slices=len(slices),
            slices=slices, duration_minutes=MARKET_MINUTES,
            expected_participation=qty / max(adv, 1),
            dark_pool_pct=dark_pct,
        )

    # ── Algorithm selection ─────────────────────────────────────────────────
    @staticmethod
    def _select_algo(urgency: str, qty: int, adv: int) -> str:
        participation = qty / max(adv, 1)
        if urgency == Urgency.CRITICAL:
            return AlgoType.IS
        if participation > 0.10:
            return AlgoType.ICEBERG
        if urgency == Urgency.LOW:
            return AlgoType.VWAP
        if urgency == Urgency.HIGH:
            return AlgoType.IS
        return AlgoType.TWAP

    # ── Comparison ──────────────────────────────────────────────────────────
    def _compare_algos(
        self, algos: Dict[str, AlgoSchedule], qty: int, adv: int,
    ) -> List[AlgoComparison]:
        results: List[AlgoComparison] = []
        participation = qty / max(adv, 1)
        for name, sched in algos.items():
            # Impact cost: η × sqrt(participation) × 10000
            cost = self.config.eta * math.sqrt(participation) * 10_000
            # Timing risk: σ × sqrt(duration/390) × 10000
            risk = self.config.sigma * math.sqrt(sched.duration_minutes / MARKET_MINUTES) * 10_000
            # IS trades faster → less risk, more cost; VWAP inverse
            if name == AlgoType.IS:
                cost *= 1.1
                risk *= 0.7
            elif name == AlgoType.VWAP:
                cost *= 0.9
                risk *= 1.1
            elif name == AlgoType.ICEBERG:
                cost *= 0.85
                risk *= 1.0

            results.append(AlgoComparison(
                algo=name,
                expected_cost_bps=cost,
                expected_risk_bps=risk,
                total_cost_bps=cost + risk,
                n_slices=sched.n_slices,
                duration_minutes=sched.duration_minutes,
                participation_rate=sched.expected_participation,
            ))
        return sorted(results, key=lambda r: r.total_cost_bps)

    # ── Benchmarking ────────────────────────────────────────────────────────
    @staticmethod
    def _benchmark(
        fill: float, arrival: float, vwap: float, close: float, side: str,
    ) -> BenchmarkResult:
        sign = 1.0 if side == "buy" else -1.0
        def _bps(ref: float) -> float:
            if ref <= 0:
                return 0.0
            return sign * (fill - ref) / ref * 10_000

        arr = _bps(arrival)
        vw = _bps(vwap)
        cl = _bps(close)
        is_bps = arr  # IS defined as vs arrival price

        benchmarks = {"arrival": abs(arr), "vwap": abs(vw), "close": abs(cl)}
        best = min(benchmarks, key=benchmarks.get)

        return BenchmarkResult(
            arrival_cost_bps=arr,
            vwap_cost_bps=vw,
            close_cost_bps=cl,
            implementation_shortfall_bps=is_bps,
            best_benchmark=best,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _urgency_duration(urgency: str) -> int:
        """Target execution window in minutes."""
        return {
            Urgency.LOW: MARKET_MINUTES,
            Urgency.MEDIUM: 180,
            Urgency.HIGH: 60,
            Urgency.CRITICAL: 15,
        }.get(urgency, 180)

    def _split_quantity(self, qty: int, n_slices: int) -> List[int]:
        """Split qty into n_slices respecting min_slice_qty."""
        if n_slices <= 0:
            return [qty]
        # Cap n_slices so each slice can be at least min_slice_qty
        effective_n = min(n_slices, max(1, qty // self.config.min_slice_qty))
        base = qty // effective_n
        remainder = qty - base * effective_n
        result = [base] * effective_n
        for i in range(remainder):
            result[i] += 1
        return result

    def _should_dark(self, qty: int, adv: int) -> bool:
        return qty >= self.config.dark_pool_threshold

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML ────────────────────────────────────────────────────────────────
    def _build_html(self, r: ExecutionResult) -> str:
        cards = self._html_cards(r)
        comp_tbl = self._html_comparison(r.comparisons)
        bench_tbl = self._html_benchmark(r.benchmark)
        sched_tbl = self._html_schedule(r.schedule)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Execution Algorithm</title>
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
.pos{{color:#4ade80}}.neg{{color:#f87171}}.sel{{background:#1e3a5f}}
</style>
</head>
<body>
<h1>Execution Algorithm Analysis</h1>
<p class="sub">Generated {r.generated_at or 'N/A'}</p>

{cards}
{comp_tbl}
{bench_tbl}
{sched_tbl}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: ExecutionResult) -> str:
        sched = r.schedule
        qty = sched.total_quantity if sched else 0
        slices = sched.n_slices if sched else 0
        dur = f"{sched.duration_minutes}m" if sched else "N/A"
        return f"""<div class="grid">
<div class="card"><div class="lbl">Selected Algo</div><div class="val">{r.selected_algo.upper()}</div></div>
<div class="card"><div class="lbl">Quantity</div><div class="val">{qty:,}</div></div>
<div class="card"><div class="lbl">Slices</div><div class="val">{slices}</div></div>
<div class="card"><div class="lbl">Duration</div><div class="val">{dur}</div></div>
<div class="card"><div class="lbl">Cost Savings</div><div class="val pos">{r.cost_savings_bps:.1f} bps</div></div>
</div>"""

    @staticmethod
    def _html_comparison(comps: List[AlgoComparison]) -> str:
        if not comps:
            return ""
        rows = ""
        for c in comps:
            rows += (
                f"<tr><td>{c.algo}</td>"
                f"<td>{c.expected_cost_bps:.1f}</td>"
                f"<td>{c.expected_risk_bps:.1f}</td>"
                f"<td><strong>{c.total_cost_bps:.1f}</strong></td>"
                f"<td>{c.n_slices}</td>"
                f"<td>{c.duration_minutes}m</td>"
                f"<td>{c.participation_rate:.2%}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Algorithm Comparison</h2>
<table>
<thead><tr><th>Algo</th><th>Cost (bps)</th><th>Risk (bps)</th><th>Total</th><th>Slices</th><th>Duration</th><th>Participation</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_benchmark(b: Optional[BenchmarkResult]) -> str:
        if not b:
            return ""
        def _cls(v: float) -> str:
            return "pos" if v <= 0 else "neg"
        return f"""<div class="sec">
<h2>Execution Quality</h2>
<table>
<thead><tr><th>Benchmark</th><th>Cost (bps)</th></tr></thead>
<tbody>
<tr><td>vs Arrival</td><td class="{_cls(b.arrival_cost_bps)}">{b.arrival_cost_bps:.1f}</td></tr>
<tr><td>vs VWAP</td><td class="{_cls(b.vwap_cost_bps)}">{b.vwap_cost_bps:.1f}</td></tr>
<tr><td>vs Close</td><td class="{_cls(b.close_cost_bps)}">{b.close_cost_bps:.1f}</td></tr>
<tr><td>Impl Shortfall</td><td>{b.implementation_shortfall_bps:.1f}</td></tr>
<tr><td>Best Benchmark</td><td class="pos">{b.best_benchmark}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_schedule(sched: Optional[AlgoSchedule]) -> str:
        if not sched:
            return ""
        rows = ""
        for s in sched.slices[:30]:
            vis = "visible" if s.is_visible else "hidden"
            dark = "dark" if s.is_dark else "lit"
            rows += (
                f"<tr><td>{s.slice_id}</td><td>{s.quantity}</td>"
                f"<td>{s.scheduled_time_min}m</td>"
                f"<td>{s.participation_rate:.3f}</td>"
                f"<td>{vis}</td><td>{dark}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Execution Schedule — {sched.algo.upper()}</h2>
<table>
<thead><tr><th>#</th><th>Qty</th><th>Time</th><th>Part. Rate</th><th>Visibility</th><th>Venue</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
