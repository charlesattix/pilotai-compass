"""
Live vs backtest comparison utilities.

Loads closed live trades from SQLite, computes real-money performance metrics,
optionally runs the backtester over the same date range, and returns a
side-by-side comparison for deviation tracking.

Functions are imported lazily by shared/deviation_tracker.py.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_live_trades(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all closed trades from the SQLite database.

    Returns:
        List of trade dicts with at minimum:
            id, ticker, entry_date, exit_date, pnl, credit, contracts
    """
    from shared.database import get_trades
    closed = (
        get_trades(status="closed_profit", path=db_path) +
        get_trades(status="closed_loss", path=db_path) +
        get_trades(status="closed_expiry", path=db_path)
    )
    return closed


def compute_live_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics from a list of closed live trades.

    Returns:
        Dict with keys: total_trades, win_rate, avg_win, avg_loss,
        profit_factor, total_pnl, start_date, end_date.
    """
    if not trades:
        return {"total_trades": 0}

    total = len(trades)
    wins = [t for t in trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0) <= 0]
    win_rate = len(wins) / total * 100 if total > 0 else 0.0
    avg_win = sum(t.get("pnl", 0) for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.get("pnl", 0) for t in losses) / len(losses) if losses else 0.0
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    gross_profit = sum(t.get("pnl", 0) for t in wins)
    gross_loss = abs(sum(t.get("pnl", 0) for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    dates = sorted(
        t.get("exit_date", "") for t in trades if t.get("exit_date")
    )
    start_date = dates[0][:10] if dates else None
    end_date = dates[-1][:10] if dates else None

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "total_pnl": round(total_pnl, 2),
        "start_date": start_date,
        "end_date": end_date,
    }


def run_backtest_for_range(
    config_path: str,
    ticker: str = "SPY",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Run the backtester over a date range and return summary metrics.

    Returns None on failure so the caller can fall back to live-only mode.
    """
    try:
        import json
        from pathlib import Path
        from backtest import Backtester

        config = json.loads(Path(config_path).read_text())
        backtester = Backtester(config)
        results = backtester.run_backtest(ticker, start, end)
        return results
    except Exception as e:
        logger.warning("run_backtest_for_range failed: %s", e)
        return None


def compare_metrics(
    live: Dict[str, Any],
    backtest: Dict[str, Any],
    warn_threshold: float = 0.15,
    fail_threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    """Compare live vs backtest metrics and return a list of comparison dicts.

    Each dict has: metric, live_value, backtest_value, deviation_pct,
    live_str, backtest_str, deviation_str, status (PASS/WARN/FAIL).
    """
    comparisons = []

    def _compare_one(name: str, live_val: float, bt_val: float, higher_is_better: bool = True):
        if bt_val == 0:
            deviation_pct = 0.0
        else:
            deviation_pct = (live_val - bt_val) / abs(bt_val) * 100

        if higher_is_better:
            bad_deviation = deviation_pct < -fail_threshold * 100
            warn_deviation = deviation_pct < -warn_threshold * 100
        else:
            bad_deviation = deviation_pct > fail_threshold * 100
            warn_deviation = deviation_pct > warn_threshold * 100

        if bad_deviation:
            status = "FAIL"
        elif warn_deviation:
            status = "WARN"
        else:
            status = "PASS"

        comparisons.append({
            "metric": name,
            "live_value": round(live_val, 4),
            "backtest_value": round(bt_val, 4),
            "deviation_pct": round(deviation_pct, 2),
            "live_str": f"{live_val:.1f}%",
            "backtest_str": f"{bt_val:.1f}%",
            "deviation_str": f"{deviation_pct:+.1f}pp",
            "status": status,
        })

    _compare_one("Win Rate", live.get("win_rate", 0), backtest.get("win_rate", 0))
    _compare_one("Profit Factor", live.get("profit_factor", 0), backtest.get("profit_factor", 0))

    return comparisons
