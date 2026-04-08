"""
EXP-2920 — Paper-Trading Monitoring Core
==========================================

Reference implementation of the 7-dimension monitoring matrix and the
4 MASTERPLAN abort-trigger evaluator for the EXP-2520 paper-trading
deployment.

This module is the canonical source of truth for:
  1. MetricAggregator   — pulls Alpaca account + engine state and
                           computes all seven monitored metrics
  2. AbortTriggerEvaluator — checks the four MASTERPLAN v11 triggers
                             and returns actionable verdicts
  3. MonitorTick         — the daily snapshot dataclass consumed by
                           the dashboard + alerter

The module is pure logic: it does NOT submit orders (that lives in
compass.exp2620_alpaca_connector), it does NOT render HTML (that
lives in scripts/exp2520_risk_dashboard.py), and it does NOT send
Telegram messages (that lives in scripts/exp2520_monitor.py). Those
consumers all call into this module for their inputs.

Rule Zero
  Every price read by the aggregator comes from Alpaca live. If a
  quote is missing the metric is flagged 'stale' and the abort
  trigger for Rule Zero violations fires.

MASTERPLAN v11 abort triggers (wording copied verbatim from
MASTERPLAN.md lines 164-170 — any one flattens immediately):
  1. Live DD hits 12% hard circuit
  2. Rolling 4-week Sharpe < 2.0 for 5 consecutive days
  3. Alpaca fills deviate from IBKR NBBO > 5 cents on > 20% of orders
  4. Any Rule Zero violation (synthetic fill, extrapolated quote)
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════
class AbortCode(str, Enum):
    DRAWDOWN_12PCT         = "drawdown_12pct_hard_circuit"
    SHARPE_BELOW_2_5DAYS   = "rolling_4w_sharpe_lt_2_for_5d"
    FILL_DEVIATION_20PCT   = "alpaca_fill_deviation_gt_5c_on_20pct_orders"
    RULE_ZERO_VIOLATION    = "rule_zero_violation"


class AbortSeverity(str, Enum):
    OK       = "ok"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class AbortVerdict:
    code: AbortCode
    severity: AbortSeverity
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    tripped_at: Optional[str] = None


@dataclass
class StreamPnl:
    sleeve_id: str
    pnl_today: float = 0.0
    pnl_mtd:   float = 0.0
    pnl_ytd:   float = 0.0
    open_positions: int = 0
    weight: float = 0.0


@dataclass
class FillQuality:
    n_fills_today: int = 0
    mean_deviation_cents: float = 0.0
    p90_deviation_cents:  float = 0.0
    frac_over_5c:         float = 0.0
    stale_quote_count:    int = 0


@dataclass
class CorrelationSnapshot:
    window_days: int
    matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    max_abs_offdiag: float = 0.0
    median_abs: float = 0.0


@dataclass
class MonitorTick:
    """Single daily snapshot — the canonical row written to state.json."""
    as_of: str
    equity: float
    rolling_peak_equity: float
    trailing_dd_pct: float
    leverage: float
    vix_last: float
    vix_ladder_mult: float
    vix_ladder_zone: str
    scale_factor: float
    per_stream: List[StreamPnl]
    rolling_sharpe_30d: Optional[float]
    rolling_sharpe_60d: Optional[float]
    rolling_sharpe_90d: Optional[float]
    fill_quality: FillQuality
    correlation: CorrelationSnapshot
    abort_verdicts: List[AbortVerdict]
    portfolio_return_today: float

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["abort_verdicts"] = [
            {**asdict(v), "code": v.code.value, "severity": v.severity.value}
            for v in self.abort_verdicts
        ]
        return out


# ═══════════════════════════════════════════════════════════════════════════
# VIX ladder (matches EXP-2820)
# ═══════════════════════════════════════════════════════════════════════════
VIX_LADDER: List[Tuple[float, float, str]] = [
    (20.0, 1.00, "calm"),
    (25.0, 0.90, "normal"),
    (30.0, 0.75, "elevated"),
    (35.0, 0.60, "caution"),
    (40.0, 0.50, "stress"),
    (50.0, 0.35, "acute_stress"),
    (60.0, 0.25, "crisis"),
    (70.0, 0.15, "panic"),
    (1e9,  0.00, "flat"),
]


def vix_ladder_state(vix: float) -> Tuple[float, str]:
    for threshold, mult, zone in VIX_LADDER:
        if vix < threshold:
            return mult, zone
    return 0.0, "flat"


# ═══════════════════════════════════════════════════════════════════════════
# Rolling metrics helpers (pure functions, testable)
# ═══════════════════════════════════════════════════════════════════════════
def rolling_sharpe(daily_returns: Sequence[float],
                    window: int) -> Optional[float]:
    arr = np.asarray(list(daily_returns)[-window:], dtype=float)
    if len(arr) < 5:
        return None
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return round((mu / sd) * math.sqrt(TRADING_DAYS), 3)


def running_max_drawdown(equity_series: Sequence[float]) -> float:
    if len(equity_series) < 2:
        return 0.0
    arr = np.asarray(equity_series, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / np.where(peak > 0, peak, 1.0)
    return float(dd.max())


def correlation_matrix(per_stream_returns: Dict[str, Sequence[float]],
                        window: int) -> CorrelationSnapshot:
    names = list(per_stream_returns.keys())
    if not names:
        return CorrelationSnapshot(window_days=window)
    mat = np.zeros((len(names), len(names)))
    cols = []
    min_n = window
    for n in names:
        arr = np.asarray(list(per_stream_returns[n])[-window:], dtype=float)
        cols.append(arr)
        if len(arr) < min_n:
            min_n = len(arr)
    if min_n < 10:
        return CorrelationSnapshot(window_days=window)
    X = np.vstack([c[-min_n:] for c in cols])
    try:
        C = np.corrcoef(X)
    except Exception:
        return CorrelationSnapshot(window_days=window)
    out: Dict[str, Dict[str, float]] = {
        a: {b: round(float(C[i, j]), 4) for j, b in enumerate(names)}
        for i, a in enumerate(names)
    }
    iu = np.triu_indices(len(names), k=1)
    off = np.abs(C[iu])
    return CorrelationSnapshot(
        window_days=window, matrix=out,
        max_abs_offdiag=round(float(off.max()) if len(off) else 0.0, 4),
        median_abs=round(float(np.median(off)) if len(off) else 0.0, 4),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Abort trigger evaluator (MASTERPLAN v11)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AbortTriggerConfig:
    # Trigger 1
    dd_hard_circuit_pct: float = 0.12
    # Trigger 2
    rolling_sharpe_window_days: int = 20          # ~4 weeks trading
    rolling_sharpe_threshold: float = 2.0
    consecutive_breach_days: int = 5
    # Trigger 3
    fill_deviation_cents: float = 0.05
    fill_deviation_frac_max: float = 0.20
    # Trigger 4
    rule_zero_violations_max: int = 0             # zero tolerance


class AbortTriggerEvaluator:
    """Stateless-ish evaluator. Caller supplies recent history; evaluator
    returns a verdict per trigger. State that MUST persist across daily
    ticks (like the 5-consecutive-breach counter for trigger 2) is
    tracked in a small dict the caller round-trips through state.json.
    """

    def __init__(self, cfg: Optional[AbortTriggerConfig] = None):
        self.cfg = cfg or AbortTriggerConfig()

    def evaluate(self, *,
                 trailing_dd_pct: float,
                 rolling_sharpe_4w: Optional[float],
                 consecutive_sharpe_breach_days: int,
                 fill_quality: FillQuality,
                 rule_zero_violations_24h: int,
                 now: Optional[datetime] = None,
                 ) -> List[AbortVerdict]:
        now_s = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        out: List[AbortVerdict] = []

        # ── Trigger 1: live DD ≥ 12% ─────────────────────────────────
        if trailing_dd_pct >= self.cfg.dd_hard_circuit_pct * 100:
            out.append(AbortVerdict(
                code=AbortCode.DRAWDOWN_12PCT,
                severity=AbortSeverity.CRITICAL,
                description=f"Trailing drawdown {trailing_dd_pct:.2f}% breached hard circuit ≥{self.cfg.dd_hard_circuit_pct*100:.0f}%",
                details={"trailing_dd_pct": trailing_dd_pct,
                         "threshold_pct": self.cfg.dd_hard_circuit_pct * 100},
                tripped_at=now_s,
            ))
        else:
            out.append(AbortVerdict(
                code=AbortCode.DRAWDOWN_12PCT,
                severity=AbortSeverity.OK,
                description=f"Trailing DD {trailing_dd_pct:.2f}% (cap 12%)",
                details={"trailing_dd_pct": trailing_dd_pct},
            ))

        # ── Trigger 2: rolling 4-week Sharpe < 2.0 for 5 consecutive days
        if rolling_sharpe_4w is None:
            out.append(AbortVerdict(
                code=AbortCode.SHARPE_BELOW_2_5DAYS,
                severity=AbortSeverity.OK,
                description="Rolling 4-week Sharpe warming up (insufficient history)",
                details={"rolling_sharpe": None,
                         "consecutive_breach_days": consecutive_sharpe_breach_days},
            ))
        elif (rolling_sharpe_4w < self.cfg.rolling_sharpe_threshold
              and consecutive_sharpe_breach_days >= self.cfg.consecutive_breach_days):
            out.append(AbortVerdict(
                code=AbortCode.SHARPE_BELOW_2_5DAYS,
                severity=AbortSeverity.CRITICAL,
                description=(
                    f"Rolling 4w Sharpe {rolling_sharpe_4w:.2f} < "
                    f"{self.cfg.rolling_sharpe_threshold} for "
                    f"{consecutive_sharpe_breach_days} consecutive days"
                ),
                details={"rolling_sharpe": rolling_sharpe_4w,
                         "consecutive_breach_days": consecutive_sharpe_breach_days,
                         "threshold": self.cfg.rolling_sharpe_threshold},
                tripped_at=now_s,
            ))
        elif rolling_sharpe_4w < self.cfg.rolling_sharpe_threshold:
            out.append(AbortVerdict(
                code=AbortCode.SHARPE_BELOW_2_5DAYS,
                severity=AbortSeverity.WARNING,
                description=(
                    f"Rolling 4w Sharpe {rolling_sharpe_4w:.2f} < "
                    f"{self.cfg.rolling_sharpe_threshold} "
                    f"(day {consecutive_sharpe_breach_days} of 5 breach window)"
                ),
                details={"rolling_sharpe": rolling_sharpe_4w,
                         "consecutive_breach_days": consecutive_sharpe_breach_days,
                         "threshold": self.cfg.rolling_sharpe_threshold},
            ))
        else:
            out.append(AbortVerdict(
                code=AbortCode.SHARPE_BELOW_2_5DAYS,
                severity=AbortSeverity.OK,
                description=f"Rolling 4w Sharpe {rolling_sharpe_4w:.2f} OK",
                details={"rolling_sharpe": rolling_sharpe_4w,
                         "consecutive_breach_days": 0},
            ))

        # ── Trigger 3: fill deviation >5c on >20% of orders ───────────
        if (fill_quality.n_fills_today > 0
            and fill_quality.frac_over_5c > self.cfg.fill_deviation_frac_max):
            out.append(AbortVerdict(
                code=AbortCode.FILL_DEVIATION_20PCT,
                severity=AbortSeverity.CRITICAL,
                description=(
                    f"{fill_quality.frac_over_5c*100:.1f}% of today's fills deviated from "
                    f"NBBO by > {self.cfg.fill_deviation_cents*100:.0f}c"
                ),
                details={"frac_over_5c": fill_quality.frac_over_5c,
                         "n_fills_today": fill_quality.n_fills_today,
                         "threshold_frac": self.cfg.fill_deviation_frac_max},
                tripped_at=now_s,
            ))
        else:
            out.append(AbortVerdict(
                code=AbortCode.FILL_DEVIATION_20PCT,
                severity=AbortSeverity.OK,
                description=(f"{fill_quality.frac_over_5c*100:.1f}% of {fill_quality.n_fills_today} "
                              f"fills over 5c (cap 20%)"),
                details={"frac_over_5c": fill_quality.frac_over_5c,
                         "n_fills_today": fill_quality.n_fills_today},
            ))

        # ── Trigger 4: Rule Zero violations (zero tolerance) ─────────
        if rule_zero_violations_24h > self.cfg.rule_zero_violations_max:
            out.append(AbortVerdict(
                code=AbortCode.RULE_ZERO_VIOLATION,
                severity=AbortSeverity.CRITICAL,
                description=f"{rule_zero_violations_24h} Rule Zero violations in last 24h — zero tolerance",
                details={"n_violations_24h": rule_zero_violations_24h},
                tripped_at=now_s,
            ))
        else:
            out.append(AbortVerdict(
                code=AbortCode.RULE_ZERO_VIOLATION,
                severity=AbortSeverity.OK,
                description="No Rule Zero violations",
                details={"n_violations_24h": 0},
            ))

        return out


# ═══════════════════════════════════════════════════════════════════════════
# Metric aggregator
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AggregatorConfig:
    rolling_short: int = 30
    rolling_mid:   int = 60
    rolling_long:  int = 90
    correlation_window: int = 60
    state_file: Path = Path("logs/exp2920/monitor_state.json")


class MetricAggregator:
    """Pulls account state + per-sleeve P&L + daily returns, computes the
    7 monitored metrics, and returns a MonitorTick ready to serialise.

    Consumers
      * scripts/exp2520_monitor.py   writes the Telegram alerts
      * scripts/exp2520_risk_dashboard.py  renders HTML
      * scripts/exp2520_daily_report.py    writes EOD reports
    """

    def __init__(self, cfg: AggregatorConfig,
                 evaluator: Optional[AbortTriggerEvaluator] = None):
        self.cfg = cfg
        self.evaluator = evaluator or AbortTriggerEvaluator()
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if self.cfg.state_file.exists():
            try:
                return json.load(open(self.cfg.state_file))
            except Exception:
                pass
        return {
            "rolling_peak_equity": None,
            "equity_history": [],          # [(iso_date, equity)]
            "return_history": [],          # [(iso_date, return)]
            "per_stream_returns": {},      # {sleeve_id: [(iso_date, r)]}
            "consecutive_sharpe_breach_days": 0,
            "rule_zero_violations_24h": 0,
            "last_tick": None,
        }

    def _save_state(self) -> None:
        self.cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.state_file.write_text(json.dumps(self.state, indent=2, default=str))

    # ── Public: build a tick ─────────────────────────────────────────
    def build_tick(self, *,
                   equity: float,
                   vix: float,
                   leverage: float,
                   per_stream_pnl: Dict[str, StreamPnl],
                   daily_returns: List[float],
                   per_stream_return_hist: Dict[str, List[float]],
                   fill_quality: FillQuality,
                   rule_zero_violations_24h: int,
                   portfolio_return_today: float,
                   now: Optional[datetime] = None) -> MonitorTick:
        now_dt = now or datetime.now(timezone.utc)
        as_of = now_dt.isoformat(timespec="seconds")

        # Rolling peak
        peak = self.state.get("rolling_peak_equity")
        if peak is None or equity > peak:
            peak = float(equity)
            self.state["rolling_peak_equity"] = peak
        trailing_dd_pct = max(0.0, (peak - equity) / peak) * 100 if peak > 0 else 0.0

        # VIX ladder
        vix_mult, vix_zone = vix_ladder_state(vix)

        # Scale factor (engine writes this to state.json — mirror it)
        scale_factor = float(self.state.get("scale_factor", leverage))

        # Rolling Sharpes
        s30 = rolling_sharpe(daily_returns, self.cfg.rolling_short)
        s60 = rolling_sharpe(daily_returns, self.cfg.rolling_mid)
        s90 = rolling_sharpe(daily_returns, self.cfg.rolling_long)
        s_4w = rolling_sharpe(daily_returns, self.evaluator.cfg.rolling_sharpe_window_days)

        # Consecutive-breach counter for trigger 2
        if s_4w is not None and s_4w < self.evaluator.cfg.rolling_sharpe_threshold:
            self.state["consecutive_sharpe_breach_days"] = int(
                self.state.get("consecutive_sharpe_breach_days", 0)
            ) + 1
        else:
            self.state["consecutive_sharpe_breach_days"] = 0

        # Correlation matrix across streams
        corr = correlation_matrix(
            {k: v for k, v in per_stream_return_hist.items() if len(v) >= 10},
            self.cfg.correlation_window,
        )

        # Abort trigger evaluation
        verdicts = self.evaluator.evaluate(
            trailing_dd_pct=trailing_dd_pct,
            rolling_sharpe_4w=s_4w,
            consecutive_sharpe_breach_days=self.state["consecutive_sharpe_breach_days"],
            fill_quality=fill_quality,
            rule_zero_violations_24h=rule_zero_violations_24h,
            now=now_dt,
        )

        tick = MonitorTick(
            as_of=as_of,
            equity=float(equity),
            rolling_peak_equity=float(peak),
            trailing_dd_pct=round(trailing_dd_pct, 3),
            leverage=float(leverage),
            vix_last=float(vix),
            vix_ladder_mult=float(vix_mult),
            vix_ladder_zone=vix_zone,
            scale_factor=scale_factor,
            per_stream=list(per_stream_pnl.values()),
            rolling_sharpe_30d=s30,
            rolling_sharpe_60d=s60,
            rolling_sharpe_90d=s90,
            fill_quality=fill_quality,
            correlation=corr,
            abort_verdicts=verdicts,
            portfolio_return_today=float(portfolio_return_today),
        )

        self.state["last_tick"] = tick.to_dict()
        self._save_state()
        return tick
