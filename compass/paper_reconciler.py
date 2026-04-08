"""
Paper Trading Reconciler V2 — compares live paper trading results against
backtest predictions with deep analytics.

Tracks six reconciliation dimensions:
  1. Signal agreement rate (backtest vs paper signal alignment)
  2. PnL deviation (live vs backtest per-trade and aggregate)
  3. Fill quality (expected vs actual fill prices)
  4. Slippage analysis (bid-ask, execution, market impact)
  5. Regime classification accuracy (predicted vs observed regime)
  6. Daily reconciliation with alerting on >10% deviation

Generates a self-contained HTML report.  This is READ-ONLY analysis — no
broker connections, no trade placement.

Usage::

    from compass.paper_reconciler import PaperReconcilerV2, ReconcilerConfig
    rec = PaperReconcilerV2(backtest_trades, paper_trades, config=ReconcilerConfig())
    result = rec.reconcile()
    PaperReconcilerV2.generate_report(result, Path("reports/reconciliation_v2.html"))
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "reconciliation_v2.html"


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class ReconcilerConfig:
    """Tolerance thresholds for reconciliation."""

    entry_price_tol_pct: float = 0.5       # 0.5% entry price tolerance
    exit_price_tol_pct: float = 0.5        # 0.5% exit price tolerance
    pnl_tol_dollars: float = 20.0          # $20 per-trade P&L tolerance
    timing_tol_minutes: float = 30.0       # 30 min entry/exit time tolerance
    deviation_alert_pct: float = 10.0      # alert when aggregate PnL deviation > 10%
    slippage_warn_bps: float = 5.0         # warn when slippage > 5 bps
    regime_match_target: float = 0.80      # target 80% regime match rate
    signal_agreement_target: float = 0.85  # target 85% signal agreement


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class SignalAgreement:
    """Signal-level agreement between backtest and paper trading."""

    total_signals: int = 0
    agreed_signals: int = 0
    disagreed_signals: int = 0
    agreement_rate: float = 0.0
    disagreements_by_type: Dict[str, int] = field(default_factory=dict)
    # type → {"direction_mismatch", "missing_in_paper", "missing_in_backtest", "confidence_divergence"}


@dataclass
class PnLDeviation:
    """Aggregate and per-trade PnL deviation analysis."""

    bt_total_pnl: float = 0.0
    pp_total_pnl: float = 0.0
    aggregate_deviation_pct: float = 0.0
    mean_per_trade_deviation: float = 0.0
    median_per_trade_deviation: float = 0.0
    std_per_trade_deviation: float = 0.0
    max_per_trade_deviation: float = 0.0
    pct_within_tolerance: float = 0.0
    daily_deviations: List[Dict[str, Any]] = field(default_factory=list)
    alert_triggered: bool = False
    alert_message: str = ""


@dataclass
class FillQuality:
    """Fill quality comparison — expected vs actual fills."""

    total_fills: int = 0
    fills_within_tolerance: int = 0
    fill_accuracy_pct: float = 0.0
    avg_entry_slippage_bps: float = 0.0
    avg_exit_slippage_bps: float = 0.0
    worst_entry_slippage_bps: float = 0.0
    worst_exit_slippage_bps: float = 0.0
    partial_fill_rate: float = 0.0
    fill_rate: float = 0.0


@dataclass
class SlippageAnalysis:
    """Detailed slippage decomposition."""

    total_slippage_dollars: float = 0.0
    avg_slippage_per_trade: float = 0.0
    slippage_as_pct_of_pnl: float = 0.0
    slippage_by_regime: Dict[str, float] = field(default_factory=dict)
    slippage_by_direction: Dict[str, float] = field(default_factory=dict)
    slippage_by_spread_type: Dict[str, float] = field(default_factory=dict)
    slippage_trend: List[float] = field(default_factory=list)  # rolling 10-trade avg


@dataclass
class RegimeAccuracy:
    """Regime classification accuracy analysis."""

    total_classified: int = 0
    correctly_classified: int = 0
    accuracy: float = 0.0
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # predicted → actual → count
    accuracy_by_regime: Dict[str, float] = field(default_factory=dict)
    regime_distribution_bt: Dict[str, int] = field(default_factory=dict)
    regime_distribution_pp: Dict[str, int] = field(default_factory=dict)


@dataclass
class TradeComparison:
    """Single trade pair comparison with full detail."""

    trade_id: str
    bt_entry_price: float
    pp_entry_price: float
    bt_exit_price: float
    pp_exit_price: float
    bt_pnl: float
    pp_pnl: float
    pnl_deviation: float
    pnl_deviation_pct: float
    entry_slippage_bps: float
    exit_slippage_bps: float
    bt_regime: str
    pp_regime: str
    regime_match: bool
    bt_signal_direction: str
    pp_signal_direction: str
    signal_match: bool
    bt_confidence: float
    pp_confidence: float
    timing_diff_minutes: float


@dataclass
class Alert:
    """Reconciliation alert."""

    severity: str          # "info", "warning", "critical"
    category: str          # "pnl_deviation", "fill_quality", "regime", "signal", "slippage"
    message: str
    value: float
    threshold: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class ReconciliationResultV2:
    """Complete reconciliation result with all six dimensions."""

    # Metadata
    reconciliation_date: str
    n_backtest_trades: int
    n_paper_trades: int
    n_matched: int

    # Six dimensions
    signal_agreement: SignalAgreement
    pnl_deviation: PnLDeviation
    fill_quality: FillQuality
    slippage_analysis: SlippageAnalysis
    regime_accuracy: RegimeAccuracy

    # Trade-level detail
    comparisons: List[TradeComparison]

    # Alerts
    alerts: List[Alert]

    # Overall score (0-100)
    reconciliation_score: float
    score_breakdown: Dict[str, float]

    # Config used
    config: ReconcilerConfig


# ── Helpers ──────────────────────────────────────────────────────────────


def _safe_pct_diff(a: float, b: float) -> float:
    """Percentage difference: (b - a) / |a| * 100. Safe for near-zero a."""
    base = abs(a) if abs(a) > 1e-9 else 1.0
    return (b - a) / base * 100.0


def _bps_diff(expected: float, actual: float) -> float:
    """Basis-point difference in price."""
    base = abs(expected) if abs(expected) > 1e-9 else 1.0
    return abs(actual - expected) / base * 10_000.0


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


# ── Matching ─────────────────────────────────────────────────────────────


def match_trades(
    bt: pd.DataFrame, pp: pd.DataFrame,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match backtest trades to paper trades by trade_id or nearest date.

    Returns (matched_pairs, unmatched_bt_indices, unmatched_pp_indices).
    """
    if "trade_id" in bt.columns and "trade_id" in pp.columns:
        bt_ids = dict(zip(bt["trade_id"], bt.index))
        pp_ids = dict(zip(pp["trade_id"], pp.index))
        common = set(bt_ids.keys()) & set(pp_ids.keys())
        pairs = [(bt_ids[k], pp_ids[k]) for k in common]
        unmatched_bt = [i for k, i in bt_ids.items() if k not in common]
        unmatched_pp = [i for k, i in pp_ids.items() if k not in common]
        return pairs, unmatched_bt, unmatched_pp

    # Fallback: nearest entry date
    bt_dates = pd.to_datetime(bt["entry_date"])
    pp_dates = pd.to_datetime(pp["entry_date"])
    used_pp: set = set()
    pairs: List[Tuple[int, int]] = []
    unmatched_bt: List[int] = []

    for bi in bt.index:
        best_j: Optional[int] = None
        best_diff = pd.Timedelta.max
        for pj in pp.index:
            if pj in used_pp:
                continue
            diff = abs(bt_dates[bi] - pp_dates[pj])
            if diff < best_diff:
                best_diff = diff
                best_j = pj
        if best_j is not None and best_diff <= pd.Timedelta(days=2):
            pairs.append((bi, best_j))
            used_pp.add(best_j)
        else:
            unmatched_bt.append(bi)

    unmatched_pp_list = [j for j in pp.index if j not in used_pp]
    return pairs, unmatched_bt, unmatched_pp_list


# ── Signal agreement ─────────────────────────────────────────────────────


def compute_signal_agreement(
    bt: pd.DataFrame, pp: pd.DataFrame,
    pairs: List[Tuple[int, int]],
    unmatched_bt: List[int],
    unmatched_pp: List[int],
) -> SignalAgreement:
    """Compute signal agreement rate between backtest and paper."""
    total = len(pairs) + len(unmatched_bt) + len(unmatched_pp)
    if total == 0:
        return SignalAgreement()

    agreed = 0
    disagreements: Dict[str, int] = {}

    for bi, pi in pairs:
        bt_row = bt.iloc[bi] if isinstance(bi, int) else bt.loc[bi]
        pp_row = pp.iloc[pi] if isinstance(pi, int) else pp.loc[pi]

        bt_dir = str(bt_row.get("direction", "short")).lower()
        pp_dir = str(pp_row.get("direction", "short")).lower()

        bt_conf = float(bt_row.get("confidence", 0.5) or 0.5)
        pp_conf = float(pp_row.get("confidence", 0.5) or 0.5)

        if bt_dir != pp_dir:
            disagreements["direction_mismatch"] = disagreements.get("direction_mismatch", 0) + 1
        elif abs(bt_conf - pp_conf) > 0.2:
            disagreements["confidence_divergence"] = disagreements.get("confidence_divergence", 0) + 1
        else:
            agreed += 1

    disagreements["missing_in_paper"] = len(unmatched_bt)
    disagreements["missing_in_backtest"] = len(unmatched_pp)

    disagreed = total - agreed
    return SignalAgreement(
        total_signals=total,
        agreed_signals=agreed,
        disagreed_signals=disagreed,
        agreement_rate=agreed / total if total > 0 else 0.0,
        disagreements_by_type={k: v for k, v in disagreements.items() if v > 0},
    )


# ── PnL deviation ───────────────────────────────────────────────────────


def compute_pnl_deviation(
    bt: pd.DataFrame, pp: pd.DataFrame,
    pairs: List[Tuple[int, int]],
    config: ReconcilerConfig,
) -> PnLDeviation:
    """Compute aggregate and per-trade PnL deviation."""
    if not pairs:
        return PnLDeviation()

    bt_pnls = []
    pp_pnls = []
    deviations = []

    for bi, pi in pairs:
        bt_row = bt.iloc[bi] if isinstance(bi, int) else bt.loc[bi]
        pp_row = pp.iloc[pi] if isinstance(pi, int) else pp.loc[pi]
        bt_pnl = float(bt_row.get("pnl", 0))
        pp_pnl = float(pp_row.get("pnl", 0))
        bt_pnls.append(bt_pnl)
        pp_pnls.append(pp_pnl)
        deviations.append(pp_pnl - bt_pnl)

    bt_total = sum(bt_pnls)
    pp_total = sum(pp_pnls)
    agg_dev = _safe_pct_diff(bt_total, pp_total) if abs(bt_total) > 1e-9 else 0.0

    within_tol = sum(1 for d in deviations if abs(d) <= config.pnl_tol_dollars)

    # Daily aggregation
    daily_devs: List[Dict[str, Any]] = []
    bt_daily: Dict[str, float] = {}
    pp_daily: Dict[str, float] = {}
    for (bi, pi), bt_p, pp_p in zip(pairs, bt_pnls, pp_pnls):
        bt_row = bt.iloc[bi] if isinstance(bi, int) else bt.loc[bi]
        date_str = str(bt_row.get("exit_date", ""))[:10]
        if date_str:
            bt_daily[date_str] = bt_daily.get(date_str, 0.0) + bt_p
            pp_daily[date_str] = pp_daily.get(date_str, 0.0) + pp_p

    for d in sorted(set(bt_daily.keys()) | set(pp_daily.keys())):
        bt_d = bt_daily.get(d, 0.0)
        pp_d = pp_daily.get(d, 0.0)
        dev_pct = _safe_pct_diff(bt_d, pp_d) if abs(bt_d) > 1e-9 else 0.0
        daily_devs.append({
            "date": d,
            "bt_pnl": round(bt_d, 2),
            "pp_pnl": round(pp_d, 2),
            "deviation_pct": round(dev_pct, 2),
        })

    alert_triggered = abs(agg_dev) > config.deviation_alert_pct
    alert_msg = ""
    if alert_triggered:
        alert_msg = (
            f"Aggregate PnL deviation {agg_dev:+.1f}% exceeds "
            f"{config.deviation_alert_pct}% threshold"
        )

    return PnLDeviation(
        bt_total_pnl=round(bt_total, 2),
        pp_total_pnl=round(pp_total, 2),
        aggregate_deviation_pct=round(agg_dev, 2),
        mean_per_trade_deviation=round(_safe_mean(deviations), 2),
        median_per_trade_deviation=round(_safe_median(deviations), 2),
        std_per_trade_deviation=round(_safe_std(deviations), 2),
        max_per_trade_deviation=round(max(abs(d) for d in deviations), 2) if deviations else 0.0,
        pct_within_tolerance=round(within_tol / len(deviations) * 100, 2) if deviations else 0.0,
        daily_deviations=daily_devs,
        alert_triggered=alert_triggered,
        alert_message=alert_msg,
    )


# ── Fill quality ─────────────────────────────────────────────────────────


def compute_fill_quality(
    bt: pd.DataFrame, pp: pd.DataFrame,
    pairs: List[Tuple[int, int]],
    config: ReconcilerConfig,
) -> FillQuality:
    """Compare expected vs actual fill prices."""
    if not pairs:
        return FillQuality()

    entry_slippages_bps = []
    exit_slippages_bps = []
    within_tol = 0

    for bi, pi in pairs:
        bt_row = bt.iloc[bi] if isinstance(bi, int) else bt.loc[bi]
        pp_row = pp.iloc[pi] if isinstance(pi, int) else pp.loc[pi]

        bt_entry = float(bt_row["entry_price"])
        pp_entry = float(pp_row["entry_price"])
        bt_exit = float(bt_row["exit_price"])
        pp_exit = float(pp_row["exit_price"])

        entry_bps = _bps_diff(bt_entry, pp_entry)
        exit_bps = _bps_diff(bt_exit, pp_exit)
        entry_slippages_bps.append(entry_bps)
        exit_slippages_bps.append(exit_bps)

        entry_pct = abs(_safe_pct_diff(bt_entry, pp_entry))
        exit_pct = abs(_safe_pct_diff(bt_exit, pp_exit))
        if entry_pct <= config.entry_price_tol_pct and exit_pct <= config.exit_price_tol_pct:
            within_tol += 1

    total = len(pairs)
    return FillQuality(
        total_fills=total,
        fills_within_tolerance=within_tol,
        fill_accuracy_pct=round(within_tol / total * 100, 2) if total else 0.0,
        avg_entry_slippage_bps=round(_safe_mean(entry_slippages_bps), 2),
        avg_exit_slippage_bps=round(_safe_mean(exit_slippages_bps), 2),
        worst_entry_slippage_bps=round(max(entry_slippages_bps), 2) if entry_slippages_bps else 0.0,
        worst_exit_slippage_bps=round(max(exit_slippages_bps), 2) if exit_slippages_bps else 0.0,
        partial_fill_rate=0.0,
        fill_rate=round(total / max(total, 1) * 100, 2),
    )


# ── Slippage analysis ───────────────────────────────────────────────────


def compute_slippage_analysis(
    bt: pd.DataFrame, pp: pd.DataFrame,
    pairs: List[Tuple[int, int]],
) -> SlippageAnalysis:
    """Decompose slippage by regime, direction, and spread type."""
    if not pairs:
        return SlippageAnalysis()

    slippages = []
    by_regime: Dict[str, List[float]] = {}
    by_direction: Dict[str, List[float]] = {}
    by_spread_type: Dict[str, List[float]] = {}

    for bi, pi in pairs:
        bt_row = bt.iloc[bi] if isinstance(bi, int) else bt.loc[bi]
        pp_row = pp.iloc[pi] if isinstance(pi, int) else pp.loc[pi]

        bt_pnl = float(bt_row.get("pnl", 0))
        pp_pnl = float(pp_row.get("pnl", 0))
        slip = bt_pnl - pp_pnl  # positive = paper underperformed (slippage cost)
        slippages.append(slip)

        regime = str(bt_row.get("regime", "unknown")).lower()
        direction = str(bt_row.get("direction", "unknown")).lower()
        spread_type = str(bt_row.get("spread_type", "unknown")).lower()

        by_regime.setdefault(regime, []).append(slip)
        by_direction.setdefault(direction, []).append(slip)
        by_spread_type.setdefault(spread_type, []).append(slip)

    total_slip = sum(slippages)
    total_pnl = sum(float(bt.iloc[bi]["pnl"]) for bi, _ in pairs) if pairs else 1.0

    # Rolling 10-trade average slippage
    arr = np.array(slippages)
    window = min(10, len(arr))
    if window > 0:
        kernel = np.ones(window) / window
        trend = np.convolve(arr, kernel, mode="valid").tolist()
    else:
        trend = []

    return SlippageAnalysis(
        total_slippage_dollars=round(total_slip, 2),
        avg_slippage_per_trade=round(_safe_mean(slippages), 2),
        slippage_as_pct_of_pnl=round(abs(total_slip) / abs(total_pnl) * 100, 2) if abs(total_pnl) > 1e-9 else 0.0,
        slippage_by_regime={k: round(_safe_mean(v), 2) for k, v in by_regime.items()},
        slippage_by_direction={k: round(_safe_mean(v), 2) for k, v in by_direction.items()},
        slippage_by_spread_type={k: round(_safe_mean(v), 2) for k, v in by_spread_type.items()},
        slippage_trend=[round(x, 2) for x in trend],
    )


# ── Regime accuracy ─────────────────────────────────────────────────────


def compute_regime_accuracy(
    bt: pd.DataFrame, pp: pd.DataFrame,
    pairs: List[Tuple[int, int]],
) -> RegimeAccuracy:
    """Compute regime classification accuracy."""
    if not pairs:
        return RegimeAccuracy()

    has_regime = "regime" in bt.columns and "regime" in pp.columns
    if not has_regime:
        return RegimeAccuracy()

    correct = 0
    total = 0
    confusion: Dict[str, Dict[str, int]] = {}
    bt_dist: Dict[str, int] = {}
    pp_dist: Dict[str, int] = {}
    correct_by_regime: Dict[str, List[bool]] = {}

    for bi, pi in pairs:
        bt_row = bt.iloc[bi] if isinstance(bi, int) else bt.loc[bi]
        pp_row = pp.iloc[pi] if isinstance(pi, int) else pp.loc[pi]

        bt_regime = str(bt_row.get("regime", "")).lower().strip()
        pp_regime = str(pp_row.get("regime", "")).lower().strip()

        if not bt_regime or not pp_regime:
            continue

        total += 1
        bt_dist[bt_regime] = bt_dist.get(bt_regime, 0) + 1
        pp_dist[pp_regime] = pp_dist.get(pp_regime, 0) + 1

        confusion.setdefault(bt_regime, {})
        confusion[bt_regime][pp_regime] = confusion[bt_regime].get(pp_regime, 0) + 1

        matched = bt_regime == pp_regime
        if matched:
            correct += 1
        correct_by_regime.setdefault(bt_regime, []).append(matched)

    accuracy_by = {
        regime: round(sum(vals) / len(vals), 4) if vals else 0.0
        for regime, vals in correct_by_regime.items()
    }

    return RegimeAccuracy(
        total_classified=total,
        correctly_classified=correct,
        accuracy=round(correct / total, 4) if total > 0 else 0.0,
        confusion_matrix=confusion,
        accuracy_by_regime=accuracy_by,
        regime_distribution_bt=bt_dist,
        regime_distribution_pp=pp_dist,
    )


# ── Trade comparison builder ────────────────────────────────────────────


def build_comparisons(
    bt: pd.DataFrame, pp: pd.DataFrame,
    pairs: List[Tuple[int, int]],
) -> List[TradeComparison]:
    """Build detailed per-trade comparisons."""
    comparisons = []

    for bi, pi in pairs:
        bt_row = bt.iloc[bi] if isinstance(bi, int) else bt.loc[bi]
        pp_row = pp.iloc[pi] if isinstance(pi, int) else pp.loc[pi]

        bt_entry = float(bt_row["entry_price"])
        pp_entry = float(pp_row["entry_price"])
        bt_exit = float(bt_row["exit_price"])
        pp_exit = float(pp_row["exit_price"])
        bt_pnl = float(bt_row["pnl"])
        pp_pnl = float(pp_row["pnl"])

        pnl_dev = pp_pnl - bt_pnl
        pnl_dev_pct = _safe_pct_diff(bt_pnl, pp_pnl)

        bt_regime = str(bt_row.get("regime", "unknown")).lower()
        pp_regime = str(pp_row.get("regime", "unknown")).lower()

        bt_dir = str(bt_row.get("direction", "short")).lower()
        pp_dir = str(pp_row.get("direction", "short")).lower()

        bt_conf = float(bt_row.get("confidence", 0.5) or 0.5)
        pp_conf = float(pp_row.get("confidence", 0.5) or 0.5)

        timing_diff = 0.0
        try:
            bt_t = pd.Timestamp(bt_row["entry_date"])
            pp_t = pd.Timestamp(pp_row["entry_date"])
            timing_diff = abs((pp_t - bt_t).total_seconds()) / 60.0
        except Exception:
            pass

        tid = str(bt_row.get("trade_id", f"pair-{bi}-{pi}"))

        comparisons.append(TradeComparison(
            trade_id=tid,
            bt_entry_price=bt_entry,
            pp_entry_price=pp_entry,
            bt_exit_price=bt_exit,
            pp_exit_price=pp_exit,
            bt_pnl=bt_pnl,
            pp_pnl=pp_pnl,
            pnl_deviation=round(pnl_dev, 2),
            pnl_deviation_pct=round(pnl_dev_pct, 2),
            entry_slippage_bps=round(_bps_diff(bt_entry, pp_entry), 2),
            exit_slippage_bps=round(_bps_diff(bt_exit, pp_exit), 2),
            bt_regime=bt_regime,
            pp_regime=pp_regime,
            regime_match=bt_regime == pp_regime,
            bt_signal_direction=bt_dir,
            pp_signal_direction=pp_dir,
            signal_match=bt_dir == pp_dir,
            bt_confidence=bt_conf,
            pp_confidence=pp_conf,
            timing_diff_minutes=round(timing_diff, 1),
        ))

    return comparisons


# ── Alert generation ────────────────────────────────────────────────────


def generate_alerts(
    pnl_dev: PnLDeviation,
    fill_qual: FillQuality,
    regime_acc: RegimeAccuracy,
    signal_agr: SignalAgreement,
    slippage: SlippageAnalysis,
    config: ReconcilerConfig,
) -> List[Alert]:
    """Generate alerts based on reconciliation results."""
    alerts: List[Alert] = []

    # PnL deviation alert
    if pnl_dev.alert_triggered:
        alerts.append(Alert(
            severity="critical",
            category="pnl_deviation",
            message=pnl_dev.alert_message,
            value=pnl_dev.aggregate_deviation_pct,
            threshold=config.deviation_alert_pct,
        ))

    # Fill quality alert
    if fill_qual.avg_entry_slippage_bps > config.slippage_warn_bps:
        alerts.append(Alert(
            severity="warning",
            category="fill_quality",
            message=f"Avg entry slippage {fill_qual.avg_entry_slippage_bps:.1f} bps "
                    f"exceeds {config.slippage_warn_bps} bps threshold",
            value=fill_qual.avg_entry_slippage_bps,
            threshold=config.slippage_warn_bps,
        ))

    # Regime accuracy alert
    if regime_acc.total_classified > 0 and regime_acc.accuracy < config.regime_match_target:
        alerts.append(Alert(
            severity="warning",
            category="regime",
            message=f"Regime accuracy {regime_acc.accuracy:.1%} below "
                    f"{config.regime_match_target:.0%} target",
            value=regime_acc.accuracy,
            threshold=config.regime_match_target,
        ))

    # Signal agreement alert
    if signal_agr.total_signals > 0 and signal_agr.agreement_rate < config.signal_agreement_target:
        alerts.append(Alert(
            severity="warning",
            category="signal",
            message=f"Signal agreement {signal_agr.agreement_rate:.1%} below "
                    f"{config.signal_agreement_target:.0%} target",
            value=signal_agr.agreement_rate,
            threshold=config.signal_agreement_target,
        ))

    # Slippage trend alert
    if slippage.slippage_as_pct_of_pnl > 5.0:
        alerts.append(Alert(
            severity="warning",
            category="slippage",
            message=f"Slippage is {slippage.slippage_as_pct_of_pnl:.1f}% of total PnL",
            value=slippage.slippage_as_pct_of_pnl,
            threshold=5.0,
        ))

    # Sort by severity
    sev_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: sev_order.get(a.severity, 9))
    return alerts


# ── Score computation ───────────────────────────────────────────────────


def compute_reconciliation_score(
    signal_agr: SignalAgreement,
    pnl_dev: PnLDeviation,
    fill_qual: FillQuality,
    regime_acc: RegimeAccuracy,
    n_matched: int,
    n_total: int,
) -> Tuple[float, Dict[str, float]]:
    """Compute 0-100 reconciliation score across six dimensions.

    Components (each 0-20, total 100):
      - match_rate: % of trades matched (0-20)
      - signal_agreement: signal alignment rate (0-20)
      - pnl_accuracy: PnL deviation proximity (0-20)
      - fill_accuracy: fill price accuracy (0-20)
      - regime_accuracy: regime classification match (0-20)
    """
    # Match rate (0-20)
    match_score = 20.0 * (n_matched / n_total) if n_total > 0 else 20.0

    # Signal agreement (0-20)
    signal_score = 20.0 * signal_agr.agreement_rate

    # PnL accuracy (0-20) — inverse of deviation
    pnl_score = 20.0 * max(0, 1.0 - abs(pnl_dev.aggregate_deviation_pct) / 100.0)

    # Fill accuracy (0-20)
    fill_score = 20.0 * (fill_qual.fill_accuracy_pct / 100.0) if fill_qual.total_fills > 0 else 20.0

    # Regime accuracy (0-20)
    regime_score = 20.0 * regime_acc.accuracy if regime_acc.total_classified > 0 else 20.0

    total = match_score + signal_score + pnl_score + fill_score + regime_score
    breakdown = {
        "match_rate": round(match_score, 2),
        "signal_agreement": round(signal_score, 2),
        "pnl_accuracy": round(pnl_score, 2),
        "fill_accuracy": round(fill_score, 2),
        "regime_accuracy": round(regime_score, 2),
    }
    return round(total, 2), breakdown


# ── Core reconciler ─────────────────────────────────────────────────────


class PaperReconcilerV2:
    """Compare live paper trading results against backtest predictions."""

    REQUIRED_COLUMNS = {"entry_price", "exit_price", "pnl", "entry_date", "exit_date"}

    def __init__(
        self,
        backtest_trades: pd.DataFrame,
        paper_trades: pd.DataFrame,
        config: Optional[ReconcilerConfig] = None,
    ):
        for name, df in [("backtest", backtest_trades), ("paper", paper_trades)]:
            missing = self.REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"{name} trades missing columns: {missing}")

        self.bt = backtest_trades.copy()
        self.pp = paper_trades.copy()
        self.config = config or ReconcilerConfig()

    def reconcile(self) -> ReconciliationResultV2:
        """Run full six-dimension reconciliation."""
        pairs, unmatched_bt, unmatched_pp = match_trades(self.bt, self.pp)

        signal_agr = compute_signal_agreement(self.bt, self.pp, pairs, unmatched_bt, unmatched_pp)
        pnl_dev = compute_pnl_deviation(self.bt, self.pp, pairs, self.config)
        fill_qual = compute_fill_quality(self.bt, self.pp, pairs, self.config)
        slippage = compute_slippage_analysis(self.bt, self.pp, pairs)
        regime_acc = compute_regime_accuracy(self.bt, self.pp, pairs)
        comparisons = build_comparisons(self.bt, self.pp, pairs)

        n_total = max(len(self.bt), len(self.pp), 1)
        score, breakdown = compute_reconciliation_score(
            signal_agr, pnl_dev, fill_qual, regime_acc, len(pairs), n_total,
        )

        alerts = generate_alerts(pnl_dev, fill_qual, regime_acc, signal_agr, slippage, self.config)

        return ReconciliationResultV2(
            reconciliation_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            n_backtest_trades=len(self.bt),
            n_paper_trades=len(self.pp),
            n_matched=len(pairs),
            signal_agreement=signal_agr,
            pnl_deviation=pnl_dev,
            fill_quality=fill_qual,
            slippage_analysis=slippage,
            regime_accuracy=regime_acc,
            comparisons=comparisons,
            alerts=alerts,
            reconciliation_score=score,
            score_breakdown=breakdown,
            config=self.config,
        )

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: ReconciliationResultV2,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ─────────────────────────────────────────────────────


def _score_color(score: float) -> str:
    if score >= 80:
        return "#3fb950"
    if score >= 60:
        return "#d29922"
    return "#f85149"


def _severity_color(severity: str) -> str:
    return {"critical": "#f85149", "warning": "#d29922", "info": "#58a6ff"}.get(severity, "#8b949e")


def _fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.2f}"


def _bar_html(label: str, value: float, max_val: float) -> str:
    pct = min(100, value / max_val * 100) if max_val > 0 else 0
    return (
        f'<div class="score-row"><span class="label">{label}</span>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%"></div></div>'
        f'<span class="value">{value:.1f}/{max_val:.0f}</span></div>'
    )


def _score_card_html(result: ReconciliationResultV2) -> str:
    color = _score_color(result.reconciliation_score)
    bd = result.score_breakdown
    return f"""
    <div class="score-card">
      <div class="big-score" style="color:{color}">{result.reconciliation_score:.0f}</div>
      <div class="score-label">Reconciliation Score V2</div>
      {_bar_html("Match Rate", bd["match_rate"], 20)}
      {_bar_html("Signal Agreement", bd["signal_agreement"], 20)}
      {_bar_html("PnL Accuracy", bd["pnl_accuracy"], 20)}
      {_bar_html("Fill Accuracy", bd["fill_accuracy"], 20)}
      {_bar_html("Regime Accuracy", bd["regime_accuracy"], 20)}
    </div>"""


def _alerts_html(alerts: List[Alert]) -> str:
    if not alerts:
        return '<div class="alert-box info">No alerts — all metrics within thresholds.</div>'
    rows = ""
    for a in alerts:
        color = _severity_color(a.severity)
        rows += (
            f'<div class="alert-box" style="border-left-color:{color}">'
            f'<span class="alert-sev" style="color:{color}">{a.severity.upper()}</span> '
            f'<span class="alert-cat">[{a.category}]</span> {a.message}</div>'
        )
    return rows


def _signal_agreement_html(sa: SignalAgreement) -> str:
    if sa.total_signals == 0:
        return "<p>No signals to compare.</p>"
    color = "#3fb950" if sa.agreement_rate >= 0.85 else "#d29922" if sa.agreement_rate >= 0.70 else "#f85149"
    disagree_rows = ""
    for dtype, count in sorted(sa.disagreements_by_type.items(), key=lambda x: -x[1]):
        disagree_rows += f'<div class="metric-row"><span>{dtype.replace("_", " ").title()}</span><span>{count}</span></div>'
    return f"""
    <div class="dimension-card">
      <div class="dim-score" style="color:{color}">{sa.agreement_rate:.1%}</div>
      <div class="dim-label">Signal Agreement Rate</div>
      <div class="metric-row"><span>Total Signals</span><span>{sa.total_signals}</span></div>
      <div class="metric-row"><span>Agreed</span><span class="good">{sa.agreed_signals}</span></div>
      <div class="metric-row"><span>Disagreed</span><span class="bad">{sa.disagreed_signals}</span></div>
      {disagree_rows}
    </div>"""


def _pnl_deviation_html(pd_: PnLDeviation) -> str:
    color = "#3fb950" if abs(pd_.aggregate_deviation_pct) < 5 else "#d29922" if abs(pd_.aggregate_deviation_pct) < 10 else "#f85149"
    alert_badge = f' <span class="alert-badge">ALERT</span>' if pd_.alert_triggered else ""
    return f"""
    <div class="dimension-card">
      <div class="dim-score" style="color:{color}">{pd_.aggregate_deviation_pct:+.1f}%{alert_badge}</div>
      <div class="dim-label">Aggregate PnL Deviation</div>
      <div class="metric-row"><span>Backtest Total PnL</span><span>{_fmt_dollar(pd_.bt_total_pnl)}</span></div>
      <div class="metric-row"><span>Paper Total PnL</span><span>{_fmt_dollar(pd_.pp_total_pnl)}</span></div>
      <div class="metric-row"><span>Mean Per-Trade Deviation</span><span>{_fmt_dollar(pd_.mean_per_trade_deviation)}</span></div>
      <div class="metric-row"><span>Median Per-Trade Deviation</span><span>{_fmt_dollar(pd_.median_per_trade_deviation)}</span></div>
      <div class="metric-row"><span>Max Per-Trade Deviation</span><span>{_fmt_dollar(pd_.max_per_trade_deviation)}</span></div>
      <div class="metric-row"><span>Within Tolerance</span><span>{pd_.pct_within_tolerance:.1f}%</span></div>
    </div>"""


def _fill_quality_html(fq: FillQuality) -> str:
    color = "#3fb950" if fq.fill_accuracy_pct >= 90 else "#d29922" if fq.fill_accuracy_pct >= 70 else "#f85149"
    return f"""
    <div class="dimension-card">
      <div class="dim-score" style="color:{color}">{fq.fill_accuracy_pct:.1f}%</div>
      <div class="dim-label">Fill Accuracy</div>
      <div class="metric-row"><span>Total Fills</span><span>{fq.total_fills}</span></div>
      <div class="metric-row"><span>Within Tolerance</span><span>{fq.fills_within_tolerance}</span></div>
      <div class="metric-row"><span>Avg Entry Slippage</span><span>{fq.avg_entry_slippage_bps:.1f} bps</span></div>
      <div class="metric-row"><span>Avg Exit Slippage</span><span>{fq.avg_exit_slippage_bps:.1f} bps</span></div>
      <div class="metric-row"><span>Worst Entry Slippage</span><span>{fq.worst_entry_slippage_bps:.1f} bps</span></div>
      <div class="metric-row"><span>Worst Exit Slippage</span><span>{fq.worst_exit_slippage_bps:.1f} bps</span></div>
    </div>"""


def _slippage_html(sa: SlippageAnalysis) -> str:
    regime_rows = ""
    for regime, avg in sorted(sa.slippage_by_regime.items()):
        regime_rows += f'<div class="metric-row"><span>{regime.title()}</span><span>{_fmt_dollar(avg)}</span></div>'
    direction_rows = ""
    for d, avg in sorted(sa.slippage_by_direction.items()):
        direction_rows += f'<div class="metric-row"><span>{d.title()}</span><span>{_fmt_dollar(avg)}</span></div>'

    trend_svg = ""
    if sa.slippage_trend:
        w, h = 500, 150
        pad = 40
        n = len(sa.slippage_trend)
        vals = sa.slippage_trend
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1.0
        points = []
        for i, v in enumerate(vals):
            x = pad + i * (w - 2 * pad) / max(n - 1, 1)
            y = h - pad - (v - mn) / rng * (h - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(points)
        trend_svg = f"""
        <svg viewBox="0 0 {w} {h}" class="chart">
          <text x="{w//2}" y="15" text-anchor="middle" class="svg-title">Slippage Trend (10-trade rolling avg)</text>
          <polyline points="{polyline}" fill="none" stroke="#58a6ff" stroke-width="2"/>
        </svg>"""

    return f"""
    <div class="dimension-card wide">
      <h3>Slippage Analysis</h3>
      <div class="metric-row"><span>Total Slippage</span><span>{_fmt_dollar(sa.total_slippage_dollars)}</span></div>
      <div class="metric-row"><span>Avg Per Trade</span><span>{_fmt_dollar(sa.avg_slippage_per_trade)}</span></div>
      <div class="metric-row"><span>As % of PnL</span><span>{sa.slippage_as_pct_of_pnl:.1f}%</span></div>
      <h4>By Regime</h4>{regime_rows}
      <h4>By Direction</h4>{direction_rows}
      {trend_svg}
    </div>"""


def _regime_accuracy_html(ra: RegimeAccuracy) -> str:
    if ra.total_classified == 0:
        return '<div class="dimension-card"><p>No regime data available.</p></div>'

    color = "#3fb950" if ra.accuracy >= 0.80 else "#d29922" if ra.accuracy >= 0.60 else "#f85149"

    confusion_html = ""
    if ra.confusion_matrix:
        all_regimes = sorted(set(
            list(ra.confusion_matrix.keys()) +
            [r for inner in ra.confusion_matrix.values() for r in inner.keys()]
        ))
        header = "".join(f"<th>{r.title()}</th>" for r in all_regimes)
        rows = ""
        for predicted in all_regimes:
            cells = ""
            for actual in all_regimes:
                val = ra.confusion_matrix.get(predicted, {}).get(actual, 0)
                cls = "good" if predicted == actual and val > 0 else ""
                cells += f'<td class="{cls}">{val}</td>'
            rows += f"<tr><td><strong>{predicted.title()}</strong></td>{cells}</tr>"
        confusion_html = f"""
        <h4>Confusion Matrix (Predicted → Actual)</h4>
        <table class="data-table"><tr><th></th>{header}</tr>{rows}</table>"""

    return f"""
    <div class="dimension-card">
      <div class="dim-score" style="color:{color}">{ra.accuracy:.1%}</div>
      <div class="dim-label">Regime Classification Accuracy</div>
      <div class="metric-row"><span>Total Classified</span><span>{ra.total_classified}</span></div>
      <div class="metric-row"><span>Correctly Classified</span><span>{ra.correctly_classified}</span></div>
      {confusion_html}
    </div>"""


def _comparison_table_html(comparisons: List[TradeComparison]) -> str:
    if not comparisons:
        return "<p>No matched trades to compare.</p>"
    rows = ""
    for c in comparisons[:100]:
        pnl_cls = "good" if c.pnl_deviation >= 0 else "bad"
        regime_icon = "&#10003;" if c.regime_match else "&#10007;"
        signal_icon = "&#10003;" if c.signal_match else "&#10007;"
        rows += f"""<tr>
          <td>{c.trade_id}</td>
          <td>{c.bt_entry_price:.2f}</td><td>{c.pp_entry_price:.2f}</td>
          <td>{c.entry_slippage_bps:.0f}</td>
          <td>{c.bt_pnl:.2f}</td><td>{c.pp_pnl:.2f}</td>
          <td class="{pnl_cls}">{_fmt_dollar(c.pnl_deviation)}</td>
          <td>{signal_icon}</td><td>{regime_icon}</td>
          <td>{c.timing_diff_minutes:.0f}m</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <tr><th>Trade</th><th>BT Entry</th><th>PP Entry</th><th>Slip (bps)</th>
          <th>BT PnL</th><th>PP PnL</th><th>PnL Δ</th>
          <th>Signal</th><th>Regime</th><th>Time Δ</th></tr>
      {rows}
    </table>"""


def _daily_deviation_table_html(daily: List[Dict[str, Any]]) -> str:
    if not daily:
        return ""
    rows = ""
    for d in daily:
        dev = d["deviation_pct"]
        cls = "bad" if abs(dev) > 10 else "warn" if abs(dev) > 5 else ""
        rows += (
            f'<tr class="{cls}"><td>{d["date"]}</td>'
            f'<td>{_fmt_dollar(d["bt_pnl"])}</td>'
            f'<td>{_fmt_dollar(d["pp_pnl"])}</td>'
            f'<td>{dev:+.1f}%</td></tr>'
        )
    return f"""
    <h3>Daily PnL Deviation</h3>
    <table class="data-table">
      <tr><th>Date</th><th>BT PnL</th><th>PP PnL</th><th>Deviation %</th></tr>
      {rows}
    </table>"""


def _build_html(result: ReconciliationResultV2) -> str:
    now = result.reconciliation_date
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Paper Trading Reconciliation V2</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3, h4 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .top-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .score-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                 padding: 24px; text-align: center; }}
  .big-score {{ font-size: 4em; font-weight: 800; }}
  .score-label {{ color: #8b949e; font-size: 1.1em; margin-bottom: 16px; }}
  .score-row {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; }}
  .score-row .label {{ width: 140px; text-align: right; color: #8b949e; font-size: 0.85em; }}
  .score-row .value {{ width: 60px; font-weight: 600; font-size: 0.85em; }}
  .bar-bg {{ flex: 1; height: 8px; background: #21262d; border-radius: 4px; }}
  .bar-fill {{ height: 100%; background: #58a6ff; border-radius: 4px; }}
  .dimension-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin: 20px 0; }}
  .dimension-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                     padding: 20px; }}
  .dimension-card.wide {{ grid-column: 1 / -1; }}
  .dim-score {{ font-size: 2.4em; font-weight: 800; text-align: center; }}
  .dim-label {{ color: #8b949e; text-align: center; margin-bottom: 12px; }}
  .metric-row {{ display: flex; justify-content: space-between; padding: 4px 0;
                 border-bottom: 1px solid #21262d; font-size: 0.9em; }}
  .good {{ color: #3fb950; }}
  .bad {{ color: #f85149; }}
  .warn {{ color: #d29922; }}
  .alert-box {{ background: #161b22; border-left: 4px solid #8b949e; border-radius: 6px;
                padding: 10px 16px; margin: 8px 0; font-size: 0.9em; }}
  .alert-box.info {{ border-left-color: #3fb950; }}
  .alert-sev {{ font-weight: 700; }}
  .alert-cat {{ color: #8b949e; }}
  .alert-badge {{ background: #f85149; color: #fff; padding: 2px 8px; border-radius: 4px;
                  font-size: 0.65em; vertical-align: middle; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.88em; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  table.data-table td:first-child, table.data-table th:first-child {{ text-align: left; }}
  .chart {{ width: 100%; max-width: 550px; margin: 12px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #21262d;
            font-size: 0.8em; color: #8b949e; }}
</style>
</head>
<body>
<h1>Paper Trading Reconciliation V2</h1>
<p class="meta">{result.n_matched} matched trades out of {result.n_backtest_trades} backtest /
   {result.n_paper_trades} paper &middot; Generated {now}</p>

<div class="top-grid">
  {_score_card_html(result)}
  <div>
    <h3>Alerts</h3>
    {_alerts_html(result.alerts)}
  </div>
</div>

<h2>Six-Dimension Analysis</h2>
<div class="dimension-grid">
  {_signal_agreement_html(result.signal_agreement)}
  {_pnl_deviation_html(result.pnl_deviation)}
  {_fill_quality_html(result.fill_quality)}
  {_regime_accuracy_html(result.regime_accuracy)}
  {_slippage_html(result.slippage_analysis)}
</div>

<h2>Trade-Level Comparison</h2>
{_comparison_table_html(result.comparisons)}

{_daily_deviation_table_html(result.pnl_deviation.daily_deviations)}

<footer>
  Generated by <code>compass/paper_reconciler.py</code> (V2) &middot; READ-ONLY analysis — no broker connections
</footer>

</body>
</html>"""
