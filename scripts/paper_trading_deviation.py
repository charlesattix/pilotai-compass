#!/usr/bin/env python3
"""Paper Trading Deviation Tracker.

Compares actual paper trade fills against PortfolioBacktester predictions
for the same date range. Tracks slippage, regime accuracy, and alignment.

Usage:
    python scripts/paper_trading_deviation.py                       # SQLite default
    python scripts/paper_trading_deviation.py --csv fills.csv       # from CSV
    python scripts/paper_trading_deviation.py --dry-run             # validate only
    python scripts/paper_trading_deviation.py --start 2026-03-01 --end 2026-03-19
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Project root on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.portfolio_backtester import PortfolioBacktester
from compass.regime import Regime, RegimeClassifier
from strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

DEFAULT_DB = str(ROOT / "data" / "pilotai_champion.db")
DEFAULT_CONFIG = str(ROOT / "configs" / "champion.json")
OUTPUT_DIR = ROOT / "results" / "paper_trading"

# Map paper-trade strategy_type → canonical strategy name
STRATEGY_TYPE_MAP = {
    "bull_put_spread": "credit_spread",
    "bear_call_spread": "credit_spread",
    "iron_condor": "iron_condor",
    "credit_spread": "credit_spread",
    "short_straddle": "straddle_strangle",
    "long_straddle": "straddle_strangle",
    "short_strangle": "straddle_strangle",
    "long_strangle": "straddle_strangle",
    "straddle_strangle": "straddle_strangle",
    "calendar_spread": "calendar_spread",
    "gamma_lotto": "gamma_lotto",
    "momentum_swing": "momentum_swing",
}

# Map paper-trade strategy_type → direction hint
DIRECTION_FROM_TYPE = {
    "bull_put_spread": "bull_put",
    "bear_call_spread": "bear_call",
    "iron_condor": "neutral",
    "short_straddle": "short",
    "long_straddle": "long",
    "short_strangle": "short",
    "long_strangle": "long",
}


# ---------------------------------------------------------------------------
# Trade loading
# ---------------------------------------------------------------------------

def load_trades_from_db(db_path: str) -> List[Dict[str, Any]]:
    """Load closed paper trades from SQLite."""
    from shared.database import get_trades

    closed = (
        get_trades(status="closed_profit", path=db_path)
        + get_trades(status="closed_loss", path=db_path)
        + get_trades(status="closed_expiry", path=db_path)
        + get_trades(status="closed_manual", path=db_path)
    )
    return closed


def load_trades_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load trades from CSV file.

    Expected columns: entry_date, ticker, strategy_type, direction,
    short_strike, long_strike, expiration, credit, contracts, pnl,
    exit_date, exit_reason
    """
    trades = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trade: Dict[str, Any] = {
                "id": f"csv_{len(trades)}",
                "ticker": row["ticker"],
                "strategy_type": row.get("strategy_type", "credit_spread"),
                "entry_date": row["entry_date"],
                "exit_date": row.get("exit_date", ""),
                "short_strike": float(row.get("short_strike", 0)),
                "long_strike": float(row.get("long_strike", 0)),
                "expiration": row.get("expiration", ""),
                "credit": float(row.get("credit", 0)),
                "contracts": int(row.get("contracts", 1)),
                "pnl": float(row.get("pnl", 0)),
                "exit_reason": row.get("exit_reason", ""),
            }
            if row.get("direction"):
                trade["direction"] = row["direction"]
            trades.append(trade)
    return trades


def _parse_date(s: str) -> Optional[datetime]:
    """Parse a date string to datetime, tolerating common formats."""
    if not s:
        return None
    s = str(s).strip().split(" ")[0].split("T")[0]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _normalize_paper_trade(t: Dict) -> Dict:
    """Normalize a paper trade to common analysis format."""
    stype = (t.get("strategy_type") or t.get("type") or "").lower()
    strategy = STRATEGY_TYPE_MAP.get(stype, stype)
    direction = t.get("direction") or DIRECTION_FROM_TYPE.get(stype, "unknown")

    entry = _parse_date(t.get("entry_date", ""))
    exit_dt = _parse_date(t.get("exit_date", ""))
    credit = float(t.get("credit", 0))
    pnl = float(t.get("pnl", 0))
    contracts = int(t.get("contracts", 1))
    short_strike = float(t.get("short_strike", 0))
    long_strike = float(t.get("long_strike", 0))
    spread_width = abs(short_strike - long_strike) if short_strike and long_strike else 0

    hold_days = (exit_dt - entry).days if entry and exit_dt else 0
    max_loss_per_unit = max(spread_width - credit, 0.01) if spread_width else max(credit, 0.01)
    return_pct = (pnl / (max_loss_per_unit * contracts * 100) * 100) if max_loss_per_unit * contracts else 0.0

    return {
        "id": t.get("id", ""),
        "source": "paper",
        "ticker": (t.get("ticker") or "").upper(),
        "strategy": strategy,
        "direction": direction,
        "entry_date": entry.strftime("%Y-%m-%d") if entry else None,
        "exit_date": exit_dt.strftime("%Y-%m-%d") if exit_dt else None,
        "credit": credit,
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 2),
        "contracts": contracts,
        "hold_days": hold_days,
        "exit_reason": t.get("exit_reason", ""),
        "short_strike": short_strike,
        "long_strike": long_strike,
    }


# ---------------------------------------------------------------------------
# Backtest replay
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict:
    """Load champion/blend config."""
    with open(config_path) as f:
        return json.load(f)


def _build_strategies(cfg: Dict) -> List[Tuple[str, Any]]:
    """Build strategy instances from config."""
    strategy_params = cfg.get("strategy_params", {})
    strategy_names = cfg.get("strategies", list(strategy_params.keys()))

    instances = []
    for name in strategy_names:
        if name not in STRATEGY_REGISTRY:
            logger.warning("Strategy %s not in registry, skipping", name)
            continue
        params = dict(strategy_params.get(name, {}))
        cls = STRATEGY_REGISTRY[name]
        instances.append((name, cls(params)))
    return instances


def run_backtest(
    config_path: str,
    start: datetime,
    end: datetime,
    tickers: List[str],
) -> List[Dict]:
    """Run PortfolioBacktester and return normalized trade list."""
    cfg = _load_config(config_path)
    strategies = _build_strategies(cfg)
    if not strategies:
        logger.error("No valid strategies in config %s", config_path)
        return []

    bt = PortfolioBacktester(
        strategies=strategies,
        tickers=tickers,
        start_date=start,
        end_date=end,
        starting_capital=100_000,
        max_positions=10,
        max_positions_per_strategy=5,
    )
    results = bt.run()
    raw_trades = results.get("trades", [])

    # Normalize backtest trades
    normalized = []
    for t in raw_trades:
        entry = _parse_date(t.get("entry_date", ""))
        exit_dt = _parse_date(t.get("exit_date", ""))
        hold_days = (exit_dt - entry).days if entry and exit_dt else 0
        normalized.append({
            "id": t.get("id", ""),
            "source": "backtest",
            "ticker": (t.get("ticker") or "").upper(),
            "strategy": t.get("strategy", ""),
            "direction": t.get("direction", ""),
            "entry_date": t.get("entry_date"),
            "exit_date": t.get("exit_date"),
            "credit": float(t.get("net_credit", 0)),
            "pnl": float(t.get("pnl", 0)),
            "return_pct": float(t.get("return_pct", 0)),
            "contracts": int(t.get("contracts", 1)),
            "hold_days": hold_days,
            "exit_reason": t.get("exit_reason", ""),
            "metadata": t.get("metadata", {}),
        })
    return normalized


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regimes(start: datetime, end: datetime) -> pd.Series:
    """Run RegimeClassifier over date range, return Series of Regime values."""
    import yfinance as yf

    warmup = 60
    fetch_start = start - timedelta(days=warmup + 30)
    fetch_end = end + timedelta(days=1)

    spy = yf.download(
        "SPY",
        start=fetch_start.strftime("%Y-%m-%d"),
        end=fetch_end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    vix = yf.download(
        "^VIX",
        start=fetch_start.strftime("%Y-%m-%d"),
        end=fetch_end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if spy.empty or vix.empty:
        logger.warning("Could not download SPY/VIX data for regime classification")
        return pd.Series(dtype=object)

    # Flatten MultiIndex columns if present
    for df in (spy, vix):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    vix_close = vix["Close"].dropna()
    classifier = RegimeClassifier()
    regime_series = classifier.classify_series(spy, vix_close)

    # Filter to requested range
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return regime_series.loc[
        (regime_series.index >= start_ts) & (regime_series.index <= end_ts)
    ]


# ---------------------------------------------------------------------------
# Trade matching
# ---------------------------------------------------------------------------

def match_trades(
    paper: List[Dict],
    backtest: List[Dict],
) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    """Match paper ↔ backtest trades by (entry_date ±1 day, ticker, strategy).

    Returns: (matched_pairs, paper_only, bt_only)
    """
    matched = []
    used_bt = set()

    for pt in paper:
        pt_date = _parse_date(pt.get("entry_date", ""))
        if not pt_date:
            continue

        best_match = None
        best_score = -1

        for i, bt in enumerate(backtest):
            if i in used_bt:
                continue
            bt_date = _parse_date(bt.get("entry_date", ""))
            if not bt_date:
                continue

            # Must match ticker and strategy
            if pt["ticker"] != bt["ticker"]:
                continue
            if pt["strategy"] != bt["strategy"]:
                continue

            # Entry date within ±1 day
            day_diff = abs((pt_date - bt_date).days)
            if day_diff > 1:
                continue

            # Score: prefer exact date match, then direction match
            score = 10 - day_diff
            if pt.get("direction") == bt.get("direction"):
                score += 5
            # Prefer closer credit
            credit_diff = abs(pt.get("credit", 0) - bt.get("credit", 0))
            score -= min(credit_diff, 5)

            if score > best_score:
                best_score = score
                best_match = i

        if best_match is not None:
            matched.append((pt, backtest[best_match]))
            used_bt.add(best_match)

    paper_only = [pt for pt in paper if not any(pt is m[0] for m in matched)]
    bt_only = [bt for i, bt in enumerate(backtest) if i not in used_bt]

    return matched, paper_only, bt_only


# ---------------------------------------------------------------------------
# Deviation computation
# ---------------------------------------------------------------------------

def _percentile(data: List[float], p: float) -> float:
    """Compute p-th percentile (0-100)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def compute_per_trade_deviations(
    matched: List[Tuple[Dict, Dict]],
    regime_series: pd.Series,
) -> List[Dict]:
    """Compute per-trade deviations for matched pairs."""
    deviations = []
    for paper, bt in matched:
        bt_credit = bt.get("credit", 0)
        credit_dev = (
            (paper["credit"] - bt_credit) / bt_credit
            if bt_credit else 0.0
        )
        pnl_dev = paper["pnl"] - bt["pnl"]
        return_dev = paper["return_pct"] - bt["return_pct"]
        hold_dev = paper["hold_days"] - bt["hold_days"]
        exit_match = _exit_reasons_match(paper.get("exit_reason", ""), bt.get("exit_reason", ""))

        # Regime on entry date
        entry_regime = "unknown"
        entry_dt = _parse_date(paper.get("entry_date", ""))
        if entry_dt and not regime_series.empty:
            ts = pd.Timestamp(entry_dt)
            if ts in regime_series.index:
                r = regime_series.loc[ts]
                entry_regime = r.value if isinstance(r, Regime) else str(r)

        # Entry slippage: positive means paper got worse fill
        entry_slippage = bt_credit - paper["credit"]

        deviations.append({
            "paper_id": paper["id"],
            "bt_id": bt["id"],
            "ticker": paper["ticker"],
            "strategy": paper["strategy"],
            "entry_date": paper["entry_date"],
            "entry_regime": entry_regime,
            "paper_credit": paper["credit"],
            "bt_credit": bt_credit,
            "credit_deviation": round(credit_dev, 4),
            "paper_pnl": paper["pnl"],
            "bt_pnl": bt["pnl"],
            "pnl_deviation": round(pnl_dev, 2),
            "paper_return_pct": paper["return_pct"],
            "bt_return_pct": bt["return_pct"],
            "return_deviation": round(return_dev, 2),
            "paper_hold_days": paper["hold_days"],
            "bt_hold_days": bt["hold_days"],
            "hold_day_deviation": hold_dev,
            "exit_reason_match": exit_match,
            "entry_slippage": round(entry_slippage, 4),
            "paper_direction": paper.get("direction", ""),
            "bt_direction": bt.get("direction", ""),
        })
    return deviations


def _exit_reasons_match(paper_reason: str, bt_reason: str) -> bool:
    """Fuzzy match exit reasons (normalize naming differences)."""
    def _norm(s: str) -> str:
        return s.lower().replace("_", "").replace(" ", "").replace("-", "")
    return _norm(paper_reason) == _norm(bt_reason)


def compute_slippage(deviations: List[Dict]) -> Dict:
    """Compute aggregate slippage analysis."""
    slippages = [d["entry_slippage"] for d in deviations]
    if not slippages:
        return {
            "entry_slippage_mean": 0.0,
            "entry_slippage_median": 0.0,
            "entry_slippage_p95": 0.0,
            "by_strategy": {},
            "by_regime": {},
            "by_month": {},
        }

    by_strategy: Dict[str, List[float]] = defaultdict(list)
    by_regime: Dict[str, List[float]] = defaultdict(list)
    by_month: Dict[str, List[float]] = defaultdict(list)

    for d in deviations:
        by_strategy[d["strategy"]].append(d["entry_slippage"])
        by_regime[d["entry_regime"]].append(d["entry_slippage"])
        entry = d.get("entry_date", "")
        if entry and len(entry) >= 7:
            by_month[entry[:7]].append(d["entry_slippage"])

    def _agg(vals: List[float]) -> Dict:
        return {
            "mean": round(sum(vals) / len(vals), 4) if vals else 0.0,
            "count": len(vals),
        }

    return {
        "entry_slippage_mean": round(sum(slippages) / len(slippages), 4),
        "entry_slippage_median": round(_percentile(slippages, 50), 4),
        "entry_slippage_p95": round(_percentile(slippages, 95), 4),
        "by_strategy": {k: _agg(v) for k, v in by_strategy.items()},
        "by_regime": {k: _agg(v) for k, v in by_regime.items()},
        "by_month": {k: _agg(v) for k, v in by_month.items()},
    }


# ---------------------------------------------------------------------------
# Regime accuracy
# ---------------------------------------------------------------------------

def compute_regime_accuracy(
    paper: List[Dict],
    backtest: List[Dict],
    regime_series: pd.Series,
) -> Dict:
    """Compute regime accuracy metrics."""
    classifier = RegimeClassifier()

    # Period distribution
    if regime_series.empty:
        distribution = {}
        transitions = 0
    else:
        summary = classifier.summarize(regime_series)
        distribution = summary.get("distribution", {})
        transitions = summary.get("transitions", 0)

    # Per-regime performance
    def _regime_for_trade(t: Dict) -> str:
        entry = _parse_date(t.get("entry_date", ""))
        if entry and not regime_series.empty:
            ts = pd.Timestamp(entry)
            if ts in regime_series.index:
                r = regime_series.loc[ts]
                return r.value if isinstance(r, Regime) else str(r)
        return "unknown"

    def _win(t: Dict) -> bool:
        return (t.get("pnl", 0) or 0) > 0

    per_regime: Dict[str, Dict] = {}
    regime_keys = set()
    for t in paper + backtest:
        regime_keys.add(_regime_for_trade(t))

    for regime in sorted(regime_keys):
        p_trades = [t for t in paper if _regime_for_trade(t) == regime]
        b_trades = [t for t in backtest if _regime_for_trade(t) == regime]

        p_wins = sum(1 for t in p_trades if _win(t))
        b_wins = sum(1 for t in b_trades if _win(t))

        per_regime[regime] = {
            "actual_trades": len(p_trades),
            "bt_trades": len(b_trades),
            "actual_wr": round(p_wins / len(p_trades) * 100, 1) if p_trades else 0.0,
            "bt_wr": round(b_wins / len(b_trades) * 100, 1) if b_trades else 0.0,
            "actual_avg_pnl": round(
                sum(t.get("pnl", 0) for t in p_trades) / len(p_trades), 2
            ) if p_trades else 0.0,
            "bt_avg_pnl": round(
                sum(t.get("pnl", 0) for t in b_trades) / len(b_trades), 2
            ) if b_trades else 0.0,
        }

    # Flags
    flags = []
    for regime, stats in per_regime.items():
        if stats["actual_trades"] >= 3 and stats["bt_trades"] >= 3:
            wr_diff = stats["actual_wr"] - stats["bt_wr"]
            if abs(wr_diff) > 10:
                flags.append(
                    f"{regime} WR {wr_diff:+.1f}pp vs backtest — "
                    + ("monitor" if abs(wr_diff) <= 20 else "investigate")
                )

    return {
        "period_distribution": distribution,
        "per_regime_performance": per_regime,
        "transitions": transitions,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_json_report(
    paper: List[Dict],
    backtest: List[Dict],
    matched: List[Tuple[Dict, Dict]],
    paper_only: List[Dict],
    bt_only: List[Dict],
    deviations: List[Dict],
    slippage: Dict,
    regime_acc: Dict,
    config_path: str,
    source: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Dict:
    """Build full JSON report."""
    n_actual = len(paper)
    n_bt = len(backtest)
    n_matched = len(matched)
    denom = max(n_actual, n_bt, 1)
    alignment = round(n_matched / denom, 2)

    if alignment >= 0.80:
        status = "ALIGNED"
    elif alignment >= 0.60:
        status = "DIVERGING"
    else:
        status = "SUSPECT"

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "date_range": {
                "start": start.strftime("%Y-%m-%d") if start else None,
                "end": end.strftime("%Y-%m-%d") if end else None,
            },
            "config_path": config_path,
            "source": source,
        },
        "summary": {
            "actual_trades": n_actual,
            "backtest_trades": n_bt,
            "matched_trades": n_matched,
            "paper_only_trades": len(paper_only),
            "bt_only_trades": len(bt_only),
            "alignment_score": alignment,
            "overall_status": status,
        },
        "per_trade_deviations": deviations,
        "slippage": slippage,
        "regime_accuracy": regime_acc,
        "unmatched": {
            "paper_only": [_trade_summary(t) for t in paper_only],
            "bt_only": [_trade_summary(t) for t in bt_only],
        },
    }


def _trade_summary(t: Dict) -> Dict:
    """Compact summary of a trade for unmatched lists."""
    return {
        "id": t.get("id", ""),
        "ticker": t.get("ticker", ""),
        "strategy": t.get("strategy", ""),
        "direction": t.get("direction", ""),
        "entry_date": t.get("entry_date", ""),
        "exit_date": t.get("exit_date", ""),
        "credit": t.get("credit", 0),
        "pnl": t.get("pnl", 0),
    }


def build_text_report(report: Dict) -> str:
    """Build human-readable text report."""
    meta = report["metadata"]
    summ = report["summary"]
    slip = report["slippage"]
    regime = report["regime_accuracy"]
    devs = report["per_trade_deviations"]

    lines = []
    lines.append(f"PAPER TRADING DEVIATION REPORT — {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("=" * 60)
    dr = meta["date_range"]
    lines.append(f"Date Range: {dr['start'] or 'N/A'} to {dr['end'] or 'N/A'}")
    lines.append(f"Config: {meta['config_path']}")
    lines.append(f"Source: {meta['source']}")
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append(f"  Actual trades:   {summ['actual_trades']:<6}Backtest trades: {summ['backtest_trades']}")
    lines.append(
        f"  Matched:         {summ['matched_trades']:<6}"
        f"Paper-only: {summ['paper_only_trades']:<5}"
        f"BT-only: {summ['bt_only_trades']}"
    )
    lines.append(
        f"  Alignment score: {summ['alignment_score']:<6}"
        f"Status: {summ['overall_status']}"
    )
    lines.append("")

    # Slippage
    lines.append("SLIPPAGE ANALYSIS")
    lines.append("  Entry slippage (actual credit vs backtest):")
    lines.append(
        f"    Mean: ${slip['entry_slippage_mean']:+.4f}    "
        f"Median: ${slip['entry_slippage_median']:+.4f}    "
        f"P95: ${slip['entry_slippage_p95']:+.4f}"
    )
    if slip.get("by_strategy"):
        lines.append("  By strategy:")
        for strat, agg in slip["by_strategy"].items():
            lines.append(f"    {strat}: ${agg['mean']:+.4f} avg ({agg['count']} trades)")
    lines.append("")

    # Regime accuracy
    lines.append("REGIME ACCURACY")
    dist = regime.get("period_distribution", {})
    if dist:
        parts = []
        for r, info in sorted(dist.items()):
            pct = info.get("pct", 0) if isinstance(info, dict) else 0
            parts.append(f"{r} {pct:.0f}%")
        lines.append(f"  Period distribution: {', '.join(parts)}")
    lines.append(f"  Transitions: {regime.get('transitions', 0)}")

    perf = regime.get("per_regime_performance", {})
    if perf:
        lines.append("  Per-regime win rates:")
        for r, stats in sorted(perf.items()):
            if stats["actual_trades"] == 0 and stats["bt_trades"] == 0:
                continue
            wr_diff = stats["actual_wr"] - stats["bt_wr"]
            flag = "PASS" if abs(wr_diff) <= 10 else f"WARN {wr_diff:+.1f}pp"
            lines.append(
                f"    {r:12s} actual {stats['actual_wr']:5.1f}% "
                f"vs backtest {stats['bt_wr']:5.1f}%  [{flag}]"
            )

    if regime.get("flags"):
        lines.append("  Flags:")
        for flag in regime["flags"]:
            lines.append(f"    - {flag}")
    lines.append("")

    # Matched trade details
    if devs:
        lines.append(f"MATCHED TRADE DETAILS ({len(devs)} trades)")
        lines.append(
            f"  {'Date':<12}{'Ticker':<6}{'Strategy':<16}{'Regime':<10}"
            f"{'Paper$':>8}{'BT$':>8}{'Slip':>8}{'PnlDev':>8}{'Exit':>5}"
        )
        lines.append("  " + "-" * 82)
        for d in devs:
            exit_mark = "Y" if d["exit_reason_match"] else "N"
            lines.append(
                f"  {d['entry_date'] or '':<12}"
                f"{d['ticker']:<6}"
                f"{d['strategy']:<16}"
                f"{d['entry_regime']:<10}"
                f"{d['paper_credit']:>8.4f}"
                f"{d['bt_credit']:>8.4f}"
                f"{d['entry_slippage']:>+8.4f}"
                f"{d['pnl_deviation']:>+8.2f}"
                f"{exit_mark:>5}"
            )
    lines.append("")

    # Unmatched
    unmatched = report.get("unmatched", {})
    paper_only = unmatched.get("paper_only", [])
    bt_only = unmatched.get("bt_only", [])

    if paper_only or bt_only:
        lines.append("UNMATCHED TRADES")
        if paper_only:
            lines.append(f"  Paper-only ({len(paper_only)}):")
            for t in paper_only:
                lines.append(
                    f"    {t['entry_date']}  {t['ticker']}  {t['strategy']}  "
                    f"credit={t['credit']:.4f}  pnl={t['pnl']:.2f}"
                )
        if bt_only:
            lines.append(f"  BT-only ({len(bt_only)}):")
            for t in bt_only:
                lines.append(
                    f"    {t['entry_date']}  {t['ticker']}  {t['strategy']}  "
                    f"credit={t['credit']:.4f}  pnl={t['pnl']:.2f}"
                )
        lines.append("")

    # Flags & action items
    all_flags = list(regime.get("flags", []))
    if paper_only:
        all_flags.append(
            f"{len(paper_only)} paper-only trades suggest signal generation divergence"
        )
    if bt_only and len(bt_only) > len(devs) * 0.5:
        all_flags.append(
            f"{len(bt_only)} BT-only trades — paper trader may be missing signals"
        )

    if all_flags:
        lines.append("FLAGS & ACTION ITEMS")
        for flag in all_flags:
            lines.append(f"  - {flag}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Deviation Tracker — compare paper fills vs backtest"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path (default: %(default)s)")
    group.add_argument("--csv", default=None, help="CSV file with trade fills")

    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config JSON path (default: %(default)s)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", default="SPY", help="Comma-separated tickers (default: SPY)")
    parser.add_argument("--dry-run", action="store_true", help="Load and validate trades only, no backtest")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determine source
    if args.csv:
        source = f"CSV ({args.csv})"
        raw_trades = load_trades_from_csv(args.csv)
    else:
        source = f"SQLite ({args.db})"
        if not os.path.exists(args.db):
            print(f"ERROR: Database not found: {args.db}")
            sys.exit(1)
        raw_trades = load_trades_from_db(args.db)

    # Normalize paper trades
    paper = [_normalize_paper_trade(t) for t in raw_trades]
    paper = [t for t in paper if t["entry_date"]]  # drop trades without entry date

    print(f"Loaded {len(paper)} closed paper trades from {source}")

    if not paper:
        print("No paper trades found. Generating empty report with regime analysis.")

    # Determine date range
    start_dt = _parse_date(args.start) if args.start else None
    end_dt = _parse_date(args.end) if args.end else None

    if paper:
        all_dates = []
        for t in paper:
            d = _parse_date(t["entry_date"])
            if d:
                all_dates.append(d)
            d = _parse_date(t.get("exit_date", ""))
            if d:
                all_dates.append(d)
        if not start_dt and all_dates:
            start_dt = min(all_dates)
        if not end_dt and all_dates:
            end_dt = max(all_dates)

    if not start_dt:
        start_dt = datetime.now() - timedelta(days=30)
    if not end_dt:
        end_dt = datetime.now()

    # Filter paper trades to date range
    paper = [
        t for t in paper
        if _parse_date(t["entry_date"])
        and start_dt <= _parse_date(t["entry_date"]) <= end_dt
    ]

    print(f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    print(f"Paper trades in range: {len(paper)}")

    if args.dry_run:
        print("\n--- DRY RUN: skipping backtest and regime classification ---")
        for t in paper[:10]:
            print(
                f"  {t['entry_date']}  {t['ticker']}  {t['strategy']:<16}"
                f"  {t['direction']:<10}  credit={t['credit']:.4f}  pnl={t['pnl']:.2f}"
            )
        if len(paper) > 10:
            print(f"  ... and {len(paper) - 10} more")
        print(f"\nDry run complete. {len(paper)} trades validated.")
        return

    # Run backtest
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    print(f"\nRunning PortfolioBacktester ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})...")
    bt_trades = run_backtest(args.config, start_dt, end_dt, tickers)
    print(f"Backtest produced {len(bt_trades)} trades")

    # Classify regimes
    print("Classifying regimes...")
    regime_series = classify_regimes(start_dt, end_dt)
    if not regime_series.empty:
        print(f"Regime data: {len(regime_series)} trading days classified")
    else:
        print("WARNING: No regime data available")

    # Match trades
    matched, paper_only, bt_only = match_trades(paper, bt_trades)
    print(f"Matched: {len(matched)}  Paper-only: {len(paper_only)}  BT-only: {len(bt_only)}")

    # Compute deviations
    deviations = compute_per_trade_deviations(matched, regime_series)
    slippage = compute_slippage(deviations)
    regime_acc = compute_regime_accuracy(paper, bt_trades, regime_series)

    # Build reports
    report = build_json_report(
        paper, bt_trades, matched, paper_only, bt_only,
        deviations, slippage, regime_acc,
        args.config, source, start_dt, end_dt,
    )
    text_report = build_text_report(report)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    json_path = OUTPUT_DIR / f"deviation_report_{date_str}.json"
    txt_path = OUTPUT_DIR / f"deviation_report_{date_str}.txt"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(txt_path, "w") as f:
        f.write(text_report)

    print(f"\nReports written to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")
    print()
    print(text_report)


if __name__ == "__main__":
    main()
