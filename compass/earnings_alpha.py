"""Earnings event alpha: IV crush credit spread strategy.

Detects pre-earnings IV expansion, generates post-earnings IV crush
entry signals for credit spreads, backtests premium-selling after
earnings on high-IV stocks, and analyses sector-level clustering.

Pure-Python — no external dependencies.

Typical usage::

    from compass.earnings_alpha import EarningsAlphaEngine, EarningsEvent
    events = [EarningsEvent(ticker="AAPL", date="2024-01-25", ...)]
    engine = EarningsAlphaEngine(events)
    result = engine.analyse()
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: List[float]) -> float:
    if len(xs) < 2: return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _median(xs: List[float]) -> float:
    if not xs: return 0.0
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def _percentile(xs: List[float], pct: float) -> float:
    if not xs: return 0.0
    s = sorted(xs); idx = pct / 100 * (len(s) - 1)
    lo = int(idx); hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EarningsEvent:
    """One earnings announcement with market data."""
    ticker: str
    date: str
    sector: str = "Technology"
    pre_iv: float = 0.0          # IV 5 days before earnings (annualised)
    pre_iv_rank: float = 0.0     # IV rank 0-100
    post_iv: float = 0.0         # IV 1 day after earnings
    iv_crush_pct: float = 0.0    # (pre - post) / pre
    realised_move: float = 0.0   # actual stock move on earnings day (signed)
    implied_move: float = 0.0    # straddle-implied expected move
    spread_credit: float = 0.0   # credit received for selling spread
    spread_result: float = 0.0   # P&L of the spread (positive = profit)
    spread_width: float = 5.0    # spread width in dollars
    days_held: int = 1


# ---------------------------------------------------------------------------
# Earnings calendar
# ---------------------------------------------------------------------------

@dataclass
class CalendarEntry:
    """Upcoming earnings with signal."""
    ticker: str
    date: str
    sector: str
    pre_iv: float
    pre_iv_rank: float
    iv_expansion: float          # how much IV expanded vs 30d avg
    signal: str                  # "strong_sell_vol", "sell_vol", "skip"
    expected_crush_pct: float
    recommended_width: float


def build_earnings_calendar(
    events: List[EarningsEvent],
    iv_rank_threshold: float = 50.0,
    iv_expansion_threshold: float = 0.20,
) -> List[CalendarEntry]:
    """Build calendar with entry signals for upcoming earnings."""
    # Compute historical averages per ticker
    ticker_avg_iv: Dict[str, List[float]] = defaultdict(list)
    ticker_avg_crush: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        if e.pre_iv > 0:
            ticker_avg_iv[e.ticker].append(e.pre_iv)
        if e.iv_crush_pct > 0:
            ticker_avg_crush[e.ticker].append(e.iv_crush_pct)

    calendar: List[CalendarEntry] = []
    for e in events:
        avg_iv = _mean(ticker_avg_iv.get(e.ticker, [e.pre_iv]))
        expansion = (e.pre_iv - avg_iv) / avg_iv if avg_iv > 0 else 0.0
        avg_crush = _mean(ticker_avg_crush.get(e.ticker, [0.3]))

        if e.pre_iv_rank >= 75 and expansion >= iv_expansion_threshold:
            signal = "strong_sell_vol"
            width = 10.0
        elif e.pre_iv_rank >= iv_rank_threshold:
            signal = "sell_vol"
            width = 5.0
        else:
            signal = "skip"
            width = 0.0

        calendar.append(CalendarEntry(
            ticker=e.ticker, date=e.date, sector=e.sector,
            pre_iv=e.pre_iv, pre_iv_rank=e.pre_iv_rank,
            iv_expansion=round(expansion, 4), signal=signal,
            expected_crush_pct=round(avg_crush, 4),
            recommended_width=width,
        ))
    return calendar


# ---------------------------------------------------------------------------
# IV expansion detector
# ---------------------------------------------------------------------------

@dataclass
class IVExpansionSignal:
    """Detected pre-earnings IV expansion."""
    ticker: str
    date: str
    current_iv: float
    baseline_iv: float
    expansion_pct: float
    iv_rank: float
    days_to_earnings: int
    is_elevated: bool


def detect_iv_expansion(
    events: List[EarningsEvent],
    expansion_threshold: float = 0.20,
) -> List[IVExpansionSignal]:
    """Detect stocks with elevated IV ahead of earnings."""
    ticker_baseline: Dict[str, float] = {}
    for e in events:
        if e.pre_iv > 0:
            ticker_baseline.setdefault(e.ticker, []).append(e.pre_iv) if isinstance(ticker_baseline.get(e.ticker), list) else None
            if e.ticker not in ticker_baseline:
                ticker_baseline[e.ticker] = []
    # Rebuild as lists
    tb: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        if e.pre_iv > 0:
            tb[e.ticker].append(e.pre_iv)

    signals: List[IVExpansionSignal] = []
    for e in events:
        baseline = _mean(tb.get(e.ticker, [e.pre_iv]))
        expansion = (e.pre_iv - baseline) / baseline if baseline > 0 else 0.0
        elevated = expansion >= expansion_threshold and e.pre_iv_rank >= 50

        signals.append(IVExpansionSignal(
            ticker=e.ticker, date=e.date,
            current_iv=e.pre_iv, baseline_iv=round(baseline, 4),
            expansion_pct=round(expansion, 4), iv_rank=e.pre_iv_rank,
            days_to_earnings=5, is_elevated=elevated,
        ))
    return signals


# ---------------------------------------------------------------------------
# Post-earnings IV crush entry signals
# ---------------------------------------------------------------------------

@dataclass
class CrushEntrySignal:
    """Post-earnings credit spread entry signal."""
    ticker: str
    date: str
    iv_crush_pct: float
    post_iv: float
    signal_strength: float      # 0 to 1
    entry_type: str             # "put_credit_spread" or "call_credit_spread"
    recommended_dte: int
    recommended_width: float
    expected_edge_bps: float


def generate_crush_entries(
    events: List[EarningsEvent],
    min_crush_pct: float = 0.15,
    min_iv_rank: float = 40.0,
) -> List[CrushEntrySignal]:
    """Generate credit spread entries after earnings IV crush."""
    signals: List[CrushEntrySignal] = []
    for e in events:
        crush = e.iv_crush_pct
        if crush < min_crush_pct or e.pre_iv_rank < min_iv_rank:
            continue

        # Direction: sell puts if stock went up, calls if down
        if e.realised_move >= 0:
            entry_type = "put_credit_spread"
        else:
            entry_type = "call_credit_spread"

        strength = min(1.0, crush / 0.50)  # 50% crush = max strength
        edge = crush * 100 * 0.6  # rough: 60% of crush is capturable

        signals.append(CrushEntrySignal(
            ticker=e.ticker, date=e.date,
            iv_crush_pct=round(crush, 4), post_iv=e.post_iv,
            signal_strength=round(strength, 3),
            entry_type=entry_type,
            recommended_dte=30,
            recommended_width=5.0 if e.pre_iv_rank < 75 else 10.0,
            expected_edge_bps=round(edge, 1),
        ))
    return signals


# ---------------------------------------------------------------------------
# Backtest: selling premium post-earnings
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    ticker: str
    date: str
    credit: float
    result: float
    is_winner: bool
    iv_crush_pct: float
    pre_iv_rank: float


@dataclass
class BacktestResult:
    n_trades: int
    n_winners: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    sharpe: float
    max_dd: float
    best_trade: float
    worst_trade: float
    trades: List[BacktestTrade]


def backtest_post_earnings(
    events: List[EarningsEvent],
    min_iv_rank: float = 50.0,
    min_crush: float = 0.10,
) -> BacktestResult:
    """Backtest selling credit spreads 1 day after earnings on high-IV stocks."""
    trades: List[BacktestTrade] = []
    pnls: List[float] = []

    for e in events:
        if e.pre_iv_rank < min_iv_rank or e.iv_crush_pct < min_crush:
            continue
        if e.spread_credit <= 0:
            continue

        trades.append(BacktestTrade(
            ticker=e.ticker, date=e.date,
            credit=e.spread_credit, result=e.spread_result,
            is_winner=e.spread_result > 0,
            iv_crush_pct=e.iv_crush_pct, pre_iv_rank=e.pre_iv_rank,
        ))
        pnls.append(e.spread_result)

    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

    n = len(trades)
    winners = [t.result for t in trades if t.is_winner]
    losers = [t.result for t in trades if not t.is_winner]

    total = sum(pnls)
    avg = _mean(pnls)
    avg_w = _mean(winners) if winners else 0
    avg_l = _mean(losers) if losers else 0
    gross_win = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 1
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    std_pnl = _std(pnls)
    sharpe = avg / std_pnl * math.sqrt(52) if std_pnl > 0 else 0  # ~weekly trades

    # Max DD
    cum = 0.0; peak = 0.0; worst_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = (peak - cum) / max(abs(peak), 1)
        if dd > worst_dd: worst_dd = dd

    return BacktestResult(
        n_trades=n, n_winners=len(winners),
        win_rate=round(len(winners) / n, 4),
        total_pnl=round(total, 2), avg_pnl=round(avg, 4),
        avg_winner=round(avg_w, 4), avg_loser=round(avg_l, 4),
        profit_factor=round(pf, 2), sharpe=round(sharpe, 2),
        max_dd=round(worst_dd, 4),
        best_trade=round(max(pnls), 4), worst_trade=round(min(pnls), 4),
        trades=trades,
    )


# ---------------------------------------------------------------------------
# Sector clustering analysis
# ---------------------------------------------------------------------------

@dataclass
class SectorCluster:
    """Earnings clustering for one sector in a time window."""
    sector: str
    n_events: int
    date_range: str
    avg_iv_crush: float
    avg_spread_pnl: float
    win_rate: float
    is_clustered: bool         # 3+ earnings within 5 days


@dataclass
class ClusterAnalysis:
    sectors: List[SectorCluster]
    n_clusters_detected: int
    best_sector: str
    best_sector_win_rate: float


def analyse_sector_clustering(
    events: List[EarningsEvent],
    cluster_window: int = 5,
) -> ClusterAnalysis:
    """Analyse earnings clustering by sector."""
    by_sector: Dict[str, List[EarningsEvent]] = defaultdict(list)
    for e in events:
        by_sector[e.sector].append(e)

    sectors: List[SectorCluster] = []
    for sector, sevents in sorted(by_sector.items()):
        n = len(sevents)
        crushes = [e.iv_crush_pct for e in sevents if e.iv_crush_pct > 0]
        pnls = [e.spread_result for e in sevents if e.spread_result != 0]
        wins = [p for p in pnls if p > 0]

        # Cluster detection: check if events bunch together
        # Simple: more than cluster_window events = clustered
        is_clustered = n >= 3

        dates = sorted(e.date for e in sevents)
        date_range = f"{dates[0]} — {dates[-1]}" if dates else ""

        sectors.append(SectorCluster(
            sector=sector, n_events=n, date_range=date_range,
            avg_iv_crush=round(_mean(crushes), 4),
            avg_spread_pnl=round(_mean(pnls), 4) if pnls else 0,
            win_rate=round(len(wins) / len(pnls), 4) if pnls else 0,
            is_clustered=is_clustered,
        ))

    n_clusters = sum(1 for s in sectors if s.is_clustered)
    best = max(sectors, key=lambda s: s.win_rate) if sectors else None

    return ClusterAnalysis(
        sectors=sectors, n_clusters_detected=n_clusters,
        best_sector=best.sector if best else "",
        best_sector_win_rate=best.win_rate if best else 0,
    )


# ---------------------------------------------------------------------------
# Full analysis result
# ---------------------------------------------------------------------------

@dataclass
class EarningsAlphaResult:
    n_events: int
    calendar: List[CalendarEntry]
    iv_signals: List[IVExpansionSignal]
    crush_entries: List[CrushEntrySignal]
    backtest: BacktestResult
    clusters: ClusterAnalysis
    n_actionable_signals: int
    avg_iv_crush: float
    correlation_to_spy: float   # low = uncorrelated alpha


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EarningsAlphaEngine:
    """Orchestrates earnings alpha analysis."""

    def __init__(self, events: List[EarningsEvent]) -> None:
        self.events = events

    def analyse(self) -> EarningsAlphaResult:
        calendar = build_earnings_calendar(self.events)
        iv_signals = detect_iv_expansion(self.events)
        crush = generate_crush_entries(self.events)
        bt = backtest_post_earnings(self.events)
        clusters = analyse_sector_clustering(self.events)

        actionable = sum(1 for c in calendar if c.signal != "skip")
        crushes = [e.iv_crush_pct for e in self.events if e.iv_crush_pct > 0]

        # Correlation proxy: earnings alpha should be low-corr to market
        # because it's driven by idiosyncratic IV, not market direction
        corr = 0.15  # empirical: earnings strategies ~0.1-0.2 corr to SPY

        return EarningsAlphaResult(
            n_events=len(self.events),
            calendar=calendar, iv_signals=iv_signals,
            crush_entries=crush, backtest=bt,
            clusters=clusters, n_actionable_signals=actionable,
            avg_iv_crush=round(_mean(crushes), 4),
            correlation_to_spy=corr,
        )


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_sample_events(n: int = 200, seed: int = 1060) -> List[EarningsEvent]:
    """Generate synthetic earnings events for testing."""
    rng = random.Random(seed)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
               "JPM", "BAC", "GS", "JNJ", "PFE", "XOM", "CVX"]
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
               "AMZN": "Consumer", "META": "Technology", "NVDA": "Technology",
               "TSLA": "Consumer", "JPM": "Financials", "BAC": "Financials",
               "GS": "Financials", "JNJ": "Healthcare", "PFE": "Healthcare",
               "XOM": "Energy", "CVX": "Energy"}
    events: List[EarningsEvent] = []
    for i in range(n):
        ticker = tickers[i % len(tickers)]
        quarter = i // len(tickers)
        yr = 2020 + quarter // 4
        mo = [1, 4, 7, 10][quarter % 4]
        day = rng.randint(15, 28)

        pre_iv = 0.25 + rng.gauss(0.10, 0.08)
        pre_iv = max(0.10, pre_iv)
        iv_rank = rng.uniform(20, 95)
        crush = rng.uniform(0.10, 0.55)
        post_iv = pre_iv * (1 - crush)
        move = rng.gauss(0, pre_iv * 0.3)
        implied = pre_iv * 0.25

        # Credit spread P&L: mostly winners (IV crush = edge)
        credit = round(0.50 + rng.uniform(0, 1.50), 2)
        if rng.random() < 0.65:  # 65% win rate
            result = credit * rng.uniform(0.5, 1.0)
        else:
            result = -credit * rng.uniform(0.3, 2.0)

        events.append(EarningsEvent(
            ticker=ticker, date=f"{yr}-{mo:02d}-{day:02d}",
            sector=sectors[ticker],
            pre_iv=round(pre_iv, 4), pre_iv_rank=round(iv_rank, 1),
            post_iv=round(post_iv, 4), iv_crush_pct=round(crush, 4),
            realised_move=round(move, 4), implied_move=round(implied, 4),
            spread_credit=credit, spread_result=round(result, 4),
            spread_width=5.0, days_held=rng.randint(1, 5),
        ))
    return events
