#!/usr/bin/env python3
"""
COMPASS ML Training Data Collection

Runs backtests (EXP-400 or EXP-401 config) and captures EVERY trade with
full market context as ML training data.

Configs:
  exp400 — CS + IC from champion.json  (original 249-trade dataset)
  exp401 — CS + SS with regime scales  (353-trade EXP-401 blend)

Output:
  compass/training_data_{config}.csv   — Chronological trade-level dataset
  compass/feature_analysis_{config}.md — Feature distributions and patterns

Usage:
    cd /home/node/openclaw/workspace/pilotai-credit-spreads
    PYTHONPATH=. python3 compass/collect_training_data.py               # exp401 (default)
    PYTHONPATH=. python3 compass/collect_training_data.py --config exp400
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.portfolio_backtester import PortfolioBacktester
from strategies import STRATEGY_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_data")

# ── Constants ──────────────────────────────────────────────────────────────
CHAMPION_PATH = ROOT / "configs" / "champion.json"
COMPASS_DIR = ROOT / "compass"
TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
YEARS = list(range(2020, 2026))

# EXP-401 regime scales (from output/regime_switching_results.json)
CS_REGIME_SCALES = {
    "regime_scale_bull": 1.0,
    "regime_scale_bear": 0.3,
    "regime_scale_high_vol": 0.3,
    "regime_scale_low_vol": 0.8,
    "regime_scale_crash": 0.0,
}

SS_REGIME_SCALES = {
    "regime_scale_bull": 1.5,
    "regime_scale_bear": 1.5,
    "regime_scale_high_vol": 2.5,
    "regime_scale_low_vol": 1.0,
    "regime_scale_crash": 0.5,
}

CS_BASE_RISK = 0.12
SS_BASE_RISK = 0.03


# ═══════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════

def load_champion_params() -> Dict:
    """Load strategy_params from champion.json."""
    with open(CHAMPION_PATH) as f:
        return json.load(f)["strategy_params"]


def _get_strategy_params(name: str, risk_override: float = None) -> Dict:
    """Get best-known params for a strategy, optionally overriding max_risk_pct."""
    champ = load_champion_params()
    if name in champ:
        params = dict(champ[name])
    else:
        cls = STRATEGY_REGISTRY[name]
        params = cls.get_default_params()
    if risk_override is not None:
        params["max_risk_pct"] = risk_override
    return params


# ═══════════════════════════════════════════════════════════════════════════
# Market data loading (no yfinance — uses backtester's data loader)
# ═══════════════════════════════════════════════════════════════════════════

def _load_full_market_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load full SPY + VIX history via PortfolioBacktester data loader.

    Creates a minimal backtester spanning 2018-2025 to get enough warmup
    for MA200 calculations on 2020 trades. No yfinance import needed —
    the backtester handles its own data sourcing.

    Returns (spy_ohlcv, vix_series).
    """
    logger.info("Loading full market data via backtester (2018-2025)...")
    bt = PortfolioBacktester(
        strategies=[],
        tickers=TICKERS,
        start_date=datetime(2018, 1, 1),
        end_date=datetime(2025, 12, 31),
        starting_capital=STARTING_CAPITAL,
    )
    bt._load_data()

    spy_data = bt._price_data.get("SPY")
    vix_series = bt._vix_series

    if spy_data is None or spy_data.empty:
        raise RuntimeError("Failed to load SPY price data via backtester")
    if vix_series is None or vix_series.empty:
        raise RuntimeError("Failed to load VIX data via backtester")

    logger.info("Market data: SPY=%d rows, VIX=%d rows", len(spy_data), len(vix_series))
    return spy_data, vix_series


# ═══════════════════════════════════════════════════════════════════════════
# Year backtests
# ═══════════════════════════════════════════════════════════════════════════

def run_year_backtest_exp400(year: int) -> Tuple[PortfolioBacktester, Dict]:
    """Run CS + IC backtest for one year (EXP-400 config)."""
    params = load_champion_params()

    cs_params = dict(params["credit_spread"])
    ic_params = dict(params["iron_condor"])

    cs_cls = STRATEGY_REGISTRY["credit_spread"]
    ic_cls = STRATEGY_REGISTRY["iron_condor"]

    bt = PortfolioBacktester(
        strategies=[
            ("credit_spread", cs_cls(cs_params)),
            ("iron_condor", ic_cls(ic_params)),
        ],
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=10,
        max_positions_per_strategy=5,
    )
    raw = bt.run()
    combined = raw.get("combined", raw)
    return bt, combined


def run_year_backtest_exp401(year: int) -> Tuple[PortfolioBacktester, Dict]:
    """Run CS + SS backtest for one year (EXP-401 config with regime scales)."""
    cs_params = _get_strategy_params("credit_spread", risk_override=CS_BASE_RISK)
    cs_params.update(CS_REGIME_SCALES)

    ss_params = _get_strategy_params("straddle_strangle", risk_override=SS_BASE_RISK)
    ss_params.update(SS_REGIME_SCALES)

    cs_cls = STRATEGY_REGISTRY["credit_spread"]
    ss_cls = STRATEGY_REGISTRY["straddle_strangle"]

    bt = PortfolioBacktester(
        strategies=[
            ("credit_spread", cs_cls(dict(cs_params))),
            ("straddle_strangle", ss_cls(dict(ss_params))),
        ],
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=10,
        max_positions_per_strategy=5,
    )
    raw = bt.run()
    combined = raw.get("combined", raw)
    return bt, combined


# ═══════════════════════════════════════════════════════════════════════════
# Trade enrichment
# ═══════════════════════════════════════════════════════════════════════════

def _prev_val(d: dict, before_ts: pd.Timestamp, default):
    """Get the most recent value in dict d with key strictly < before_ts."""
    keys = [k for k in d if k < before_ts]
    return d[max(keys)] if keys else default


def _compute_vix_percentile(vix_series: pd.Series, date_ts: pd.Timestamp, window: int) -> float:
    """Compute VIX percentile rank over trailing window days."""
    if vix_series is None or date_ts not in vix_series.index:
        return 50.0
    hist = vix_series.loc[vix_series.index <= date_ts].tail(window)
    if len(hist) < 10:
        return 50.0
    current = float(vix_series.loc[date_ts])
    below = (hist < current).sum()
    return round((below / len(hist)) * 100, 2)


def _compute_ma(closes: pd.Series, date_ts: pd.Timestamp, period: int) -> Optional[float]:
    """Compute moving average at a specific date."""
    hist = closes.loc[closes.index <= date_ts]
    if len(hist) < period:
        return None
    return float(hist.tail(period).mean())


def _compute_ma_slope(closes: pd.Series, date_ts: pd.Timestamp, ma_period: int, lookback: int = 20) -> Optional[float]:
    """Compute annualized slope of a moving average (%)."""
    hist = closes.loc[closes.index <= date_ts]
    if len(hist) < ma_period + lookback:
        return None
    ma_now = float(hist.tail(ma_period).mean())
    earlier = hist.iloc[:-lookback]
    if len(earlier) < ma_period:
        return None
    ma_then = float(earlier.tail(ma_period).mean())
    if ma_then == 0:
        return 0.0
    pct_change = (ma_now - ma_then) / ma_then
    annualized = pct_change * (252 / lookback) * 100
    return round(annualized, 4)


def _compute_returns_vol(closes: pd.Series, date_ts: pd.Timestamp, window: int) -> Optional[float]:
    """Compute returns-based realized vol (annualized %) over trailing window."""
    hist = closes.loc[closes.index <= date_ts]
    if len(hist) < window + 1:
        return None
    returns = hist.pct_change().dropna().tail(window)
    if len(returns) < window:
        return None
    return round(float(returns.std() * math.sqrt(252) * 100), 4)


def _classify_strategy_type(strat_name: str) -> str:
    """Map strategy name to short code for training data."""
    lower = strat_name.lower()
    if "iron" in lower or "condor" in lower:
        return "IC"
    elif "straddle" in lower or "strangle" in lower:
        return "SS"
    else:
        return "CS"


def enrich_trades(
    bt: PortfolioBacktester,
    year: int,
    spy_closes: pd.Series = None,
    vix_series: pd.Series = None,
) -> List[Dict]:
    """Enrich closed trades with full market context.

    Args:
        bt: Completed backtester with closed_trades
        year: Year label for the trades
        spy_closes: Optional full SPY close series (for MA200 warmup).
                    If not provided, uses bt._price_data["SPY"]["Close"].
        vix_series: Optional full VIX series. If not provided, uses bt._vix_series.
    """
    enriched = []

    if spy_closes is None:
        spy_data = bt._price_data.get("SPY")
        if spy_data is None or spy_data.empty:
            logger.warning("No SPY price data for year %d", year)
            return enriched
        spy_closes = spy_data["Close"]

    if vix_series is None:
        vix_series = bt._vix_series

    # Sort trades by entry date (chronological)
    sorted_trades = sorted(bt.closed_trades, key=lambda t: t.entry_date or datetime.min)

    prev_entry_date = None

    for trade in sorted_trades:
        if trade.entry_date is None or trade.exit_date is None:
            continue

        entry_dt = trade.entry_date
        exit_dt = trade.exit_date
        entry_ts = pd.Timestamp(entry_dt)

        # Strategy type
        strategy_type = _classify_strategy_type(trade.strategy_name)

        # Spread type from metadata
        spread_type = (trade.metadata or {}).get("spread_type", "unknown")

        # DTE at entry
        dte_at_entry = None
        if trade.legs:
            exp = trade.legs[0].expiration
            if exp:
                dte_at_entry = (exp - entry_dt).days

        # Hold duration
        hold_days = (exit_dt - entry_dt).days

        # Entry price context
        spy_price = None
        if entry_ts in spy_closes.index:
            spy_price = float(spy_closes.loc[entry_ts])
        elif len(spy_closes.loc[spy_closes.index <= entry_ts]) > 0:
            spy_price = float(spy_closes.loc[spy_closes.index <= entry_ts].iloc[-1])

        # Moving averages
        ma_20 = _compute_ma(spy_closes, entry_ts, 20)
        ma_50 = _compute_ma(spy_closes, entry_ts, 50)
        ma_80 = _compute_ma(spy_closes, entry_ts, 80)
        ma_200 = _compute_ma(spy_closes, entry_ts, 200)

        # Distance from MA (%)
        dist_ma_20 = round((spy_price - ma_20) / ma_20 * 100, 4) if spy_price and ma_20 else None
        dist_ma_50 = round((spy_price - ma_50) / ma_50 * 100, 4) if spy_price and ma_50 else None
        dist_ma_80 = round((spy_price - ma_80) / ma_80 * 100, 4) if spy_price and ma_80 else None
        dist_ma_200 = round((spy_price - ma_200) / ma_200 * 100, 4) if spy_price and ma_200 else None

        # MA slopes (annualized %)
        ma_20_slope = _compute_ma_slope(spy_closes, entry_ts, 20)
        ma_50_slope = _compute_ma_slope(spy_closes, entry_ts, 50)

        # VIX at entry
        vix_val = None
        if vix_series is not None and entry_ts in vix_series.index:
            vix_val = float(vix_series.loc[entry_ts])
        elif vix_series is not None:
            prior = vix_series.loc[vix_series.index <= entry_ts]
            if len(prior) > 0:
                vix_val = float(prior.iloc[-1])

        # VIX percentile ranks
        vix_pctile_20 = _compute_vix_percentile(vix_series, entry_ts, 20)
        vix_pctile_50 = _compute_vix_percentile(vix_series, entry_ts, 50)
        vix_pctile_100 = _compute_vix_percentile(vix_series, entry_ts, 100)

        # IV rank
        iv_rank = bt._iv_rank_by_date.get(entry_ts, None)
        if iv_rank is None:
            prior_keys = [k for k in bt._iv_rank_by_date if k <= entry_ts]
            if prior_keys:
                iv_rank = bt._iv_rank_by_date[max(prior_keys)]

        # Realized vol (ATR-based, from backtester)
        rv_map = bt._realized_vol_by_date.get("SPY", {})
        realized_vol_atr = rv_map.get(entry_ts, None)
        if realized_vol_atr is None:
            prior_keys = [k for k in rv_map if k <= entry_ts]
            if prior_keys:
                realized_vol_atr = rv_map[max(prior_keys)]

        # Returns-based realized vol (5/10/20 day)
        rv_5d = _compute_returns_vol(spy_closes, entry_ts, 5)
        rv_10d = _compute_returns_vol(spy_closes, entry_ts, 10)
        rv_20d = _compute_returns_vol(spy_closes, entry_ts, 20)

        # RSI
        rsi_map = bt._rsi_by_date.get("SPY", {})
        rsi_val = rsi_map.get(entry_ts, None)
        if rsi_val is None:
            prior_keys = [k for k in rsi_map if k <= entry_ts]
            if prior_keys:
                rsi_val = rsi_map[max(prior_keys)]

        # Regime at entry
        regime = (trade.metadata or {}).get("regime", None)

        # Day of week (0=Mon, 4=Fri)
        day_of_week = entry_dt.weekday()

        # Days since last trade
        days_since_last = None
        if prev_entry_date is not None:
            days_since_last = (entry_dt - prev_entry_date).days
        prev_entry_date = entry_dt

        # Premium collected, spread width
        net_credit = trade.net_credit
        max_loss = trade.max_loss_per_unit

        # Spread width from legs
        spread_width = None
        if trade.legs and len(trade.legs) >= 2:
            strikes = sorted([leg.strike for leg in trade.legs])
            spread_width = strikes[-1] - strikes[0]
            # For IC with 4 legs, take max width between pairs
            if len(trade.legs) == 4:
                spread_width = max(
                    abs(trade.legs[0].strike - trade.legs[1].strike),
                    abs(trade.legs[2].strike - trade.legs[3].strike),
                )

        # Short strike delta proxy (OTM distance)
        short_strike = None
        if trade.legs:
            short_legs = [l for l in trade.legs if "short" in l.leg_type.value]
            if short_legs:
                short_strike = short_legs[0].strike

        otm_pct = None
        if short_strike and spy_price:
            otm_pct = round(abs(spy_price - short_strike) / spy_price * 100, 4)

        # Contracts
        contracts = trade.contracts

        # Momentum (10-day return)
        mom_10d = None
        hist = spy_closes.loc[spy_closes.index <= entry_ts]
        if len(hist) >= 11:
            mom_10d = round((float(hist.iloc[-1]) - float(hist.iloc[-11])) / float(hist.iloc[-11]) * 100, 4)

        # 5-day return
        mom_5d = None
        if len(hist) >= 6:
            mom_5d = round((float(hist.iloc[-1]) - float(hist.iloc[-6])) / float(hist.iloc[-6]) * 100, 4)

        # OUTCOME
        pnl = trade.realized_pnl
        max_risk = max_loss * contracts * 100 if max_loss and contracts else 0
        return_pct = round((pnl / max_risk) * 100, 4) if max_risk > 0 else 0
        win = 1 if pnl > 0 else 0

        row = {
            # Identity
            "entry_date": entry_dt.strftime("%Y-%m-%d"),
            "exit_date": exit_dt.strftime("%Y-%m-%d"),
            "year": year,
            "strategy_type": strategy_type,
            "spread_type": spread_type,

            # Timing
            "dte_at_entry": dte_at_entry,
            "hold_days": hold_days,
            "day_of_week": day_of_week,
            "days_since_last_trade": days_since_last,

            # Regime & signals
            "regime": regime,
            "rsi_14": round(rsi_val, 2) if rsi_val is not None else None,
            "momentum_5d_pct": mom_5d,
            "momentum_10d_pct": mom_10d,

            # VIX
            "vix": round(vix_val, 2) if vix_val is not None else None,
            "vix_percentile_20d": vix_pctile_20,
            "vix_percentile_50d": vix_pctile_50,
            "vix_percentile_100d": vix_pctile_100,
            "iv_rank": round(iv_rank, 2) if iv_rank is not None else None,

            # Price & MAs
            "spy_price": round(spy_price, 2) if spy_price else None,
            "dist_from_ma20_pct": dist_ma_20,
            "dist_from_ma50_pct": dist_ma_50,
            "dist_from_ma80_pct": dist_ma_80,
            "dist_from_ma200_pct": dist_ma_200,
            "ma20_slope_ann_pct": ma_20_slope,
            "ma50_slope_ann_pct": ma_50_slope,

            # Volatility
            "realized_vol_atr20": round(realized_vol_atr * 100, 2) if realized_vol_atr else None,
            "realized_vol_5d": rv_5d,
            "realized_vol_10d": rv_10d,
            "realized_vol_20d": rv_20d,

            # Trade structure
            "net_credit": round(net_credit, 4) if net_credit else None,
            "spread_width": spread_width,
            "max_loss_per_unit": round(max_loss, 4) if max_loss else None,
            "short_strike": short_strike,
            "otm_pct": otm_pct,
            "contracts": contracts,

            # Outcome
            "exit_reason": trade.exit_reason,
            "pnl": round(pnl, 2),
            "return_pct": return_pct,
            "win": win,
        }

        enriched.append(row)

    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# Feature analysis report
# ═══════════════════════════════════════════════════════════════════════════

def generate_feature_analysis(df: pd.DataFrame, config_name: str = "exp401") -> str:
    """Generate markdown feature analysis report."""
    lines = [
        f"# {config_name.upper()} — Feature Analysis",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total trades:** {len(df)}",
        f"**Date range:** {df['entry_date'].min()} to {df['entry_date'].max()}",
        f"**Overall win rate:** {df['win'].mean() * 100:.1f}%",
        f"**Avg return per trade:** {df['return_pct'].mean():.2f}%",
        "",
        "---",
        "",
        "## 1. Feature Distributions",
        "",
    ]

    numeric_cols = [
        "dte_at_entry", "hold_days", "vix", "iv_rank",
        "spy_price", "dist_from_ma20_pct", "dist_from_ma80_pct",
        "dist_from_ma200_pct", "realized_vol_20d",
        "rsi_14", "momentum_10d_pct", "net_credit", "spread_width",
        "otm_pct", "return_pct",
    ]

    lines.append("| Feature | Count | Mean | Std | Min | 25% | 50% | 75% | Max |")
    lines.append("|---------|-------|------|-----|-----|-----|-----|-----|-----|")
    for col in numeric_cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                desc = s.describe()
                lines.append(
                    f"| {col} | {int(desc['count'])} | {desc['mean']:.2f} | {desc['std']:.2f} | "
                    f"{desc['min']:.2f} | {desc['25%']:.2f} | {desc['50%']:.2f} | "
                    f"{desc['75%']:.2f} | {desc['max']:.2f} |"
                )

    # Strategy breakdown
    lines.extend(["", "## 2. Strategy Breakdown", ""])
    for strat in df["strategy_type"].unique():
        subset = df[df["strategy_type"] == strat]
        lines.append(f"### {strat}")
        lines.append(f"- Trades: {len(subset)}")
        lines.append(f"- Win rate: {subset['win'].mean() * 100:.1f}%")
        lines.append(f"- Avg return: {subset['return_pct'].mean():.2f}%")
        lines.append(f"- Avg hold days: {subset['hold_days'].mean():.1f}")
        lines.append("")

    # Win rate by regime
    lines.extend([
        "## 3. Win Rate by Regime", "",
        "| Regime | Trades | Win Rate | Avg Return |",
        "|--------|--------|----------|------------|",
    ])
    for regime in sorted(df["regime"].dropna().unique()):
        subset = df[df["regime"] == regime]
        lines.append(
            f"| {regime} | {len(subset)} | {subset['win'].mean() * 100:.1f}% | "
            f"{subset['return_pct'].mean():.2f}% |"
        )

    # Win rate by year
    lines.extend([
        "", "## 4. Win Rate by Year", "",
        "| Year | Trades | Win Rate | Avg Return | Total PnL |",
        "|------|--------|----------|------------|-----------|",
    ])
    for year in sorted(df["year"].unique()):
        subset = df[df["year"] == year]
        lines.append(
            f"| {year} | {len(subset)} | {subset['win'].mean() * 100:.1f}% | "
            f"{subset['return_pct'].mean():.2f}% | ${subset['pnl'].sum():,.0f} |"
        )

    # Feature correlations with outcome
    lines.extend([
        "", "## 5. Feature Correlations with Win/Loss", "",
        "| Feature | Correlation | p-value (approx) |",
        "|---------|-------------|------------------|",
    ])
    corr_cols = [
        "vix", "iv_rank", "rsi_14", "dte_at_entry",
        "dist_from_ma20_pct", "dist_from_ma80_pct", "dist_from_ma200_pct",
        "realized_vol_20d", "momentum_10d_pct", "vix_percentile_50d",
        "otm_pct", "net_credit", "day_of_week",
    ]
    for col in corr_cols:
        if col in df.columns:
            valid = df[[col, "win"]].dropna()
            if len(valid) > 5:
                corr = valid[col].corr(valid["win"])
                n = len(valid)
                if abs(corr) < 1:
                    t_stat = corr * math.sqrt(n - 2) / math.sqrt(1 - corr**2)
                    p_approx = "< 0.05" if abs(t_stat) > 2.0 else ">= 0.05"
                else:
                    p_approx = "< 0.01"
                lines.append(f"| {col} | {corr:+.4f} | {p_approx} |")

    lines.extend(["", "---", "", f"*Generated by compass/collect_training_data.py — {config_name}*"])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset merge + dedup
# ═══════════════════════════════════════════════════════════════════════════

DEDUP_KEYS = ["entry_date", "exit_date", "strategy_type", "spread_type"]

EXP400_PATH = ROOT / "ml" / "training_data.csv"
EXP401_PATH = COMPASS_DIR / "training_data_exp401.csv"
COMBINED_PATH = COMPASS_DIR / "training_data_combined.csv"


def merge_datasets() -> pd.DataFrame:
    """Merge EXP-400 + EXP-401 datasets into a combined, deduplicated dataset.

    Deduplicates on (entry_date, exit_date, strategy_type, spread_type).
    When duplicates exist, EXP-401 rows are preferred (newer config).

    Returns the merged DataFrame.
    """
    dfs = []

    for path, label in [(EXP400_PATH, "exp400"), (EXP401_PATH, "exp401")]:
        if not path.exists():
            logger.warning("Dataset not found: %s — skipping", path)
            continue
        df = pd.read_csv(path)
        df["_source"] = label
        logger.info("Loaded %s: %d trades", label, len(df))
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No source datasets found for merge")

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)

    # Dedup: keep last (exp401 is appended second, so it wins)
    combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="last")
    combined = combined.sort_values("entry_date").reset_index(drop=True)

    dupes_removed = before - len(combined)
    logger.info("Merged: %d total, %d duplicates removed, %d final",
                before, dupes_removed, len(combined))

    # Drop internal column
    source_counts = combined["_source"].value_counts().to_dict()
    combined = combined.drop(columns=["_source"])

    # Save
    combined.to_csv(COMBINED_PATH, index=False)
    logger.info("Saved combined dataset to %s", COMBINED_PATH)

    # Generate analysis for combined
    analysis = generate_feature_analysis(combined, "combined")
    analysis_path = COMPASS_DIR / "feature_analysis_combined.md"
    with open(analysis_path, "w") as f:
        f.write(analysis)
    logger.info("Saved combined feature analysis to %s", analysis_path)

    # Summary
    logger.info("Source breakdown: %s", source_counts)
    logger.info("Strategy split: %s", combined["strategy_type"].value_counts().to_dict())
    logger.info("Year range: %s to %s", combined["entry_date"].min(), combined["entry_date"].max())
    logger.info("Overall win rate: %.1f%%", combined["win"].mean() * 100)

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Collect ML training data")
    parser.add_argument("--config", choices=["exp400", "exp401"], default="exp401",
                        help="Backtest config: exp400 (CS+IC) or exp401 (CS+SS regime)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge EXP-400 + EXP-401 into combined dataset (no backtest)")
    args = parser.parse_args()

    if args.merge:
        merge_datasets()
        return

    config_name = args.config
    t0 = time.time()

    logger.info("COMPASS ML Training Data Collection")
    logger.info("Config: %s", config_name)

    if config_name == "exp400":
        run_fn = run_year_backtest_exp400
        label = "EXP-400 (CS + IC)"
    else:
        run_fn = run_year_backtest_exp401
        label = "EXP-401 (CS + SS regime-adaptive)"

    logger.info("Running %s on SPY, %d-%d", label, YEARS[0], YEARS[-1])

    # Load full market data once for accurate MA200 + VIX percentiles
    full_spy, full_vix = _load_full_market_data()
    full_spy_closes = full_spy["Close"]

    all_trades = []

    for year in YEARS:
        logger.info("=" * 60)
        logger.info("Year %d", year)
        logger.info("=" * 60)

        bt, results = run_fn(year)

        # Enrich with full history (for MA200 warmup)
        trades = enrich_trades(bt, year, spy_closes=full_spy_closes, vix_series=full_vix)
        logger.info(
            "Year %d: %d trades, return=%.2f%%, WR=%.1f%%",
            year,
            len(trades),
            results.get("return_pct", 0),
            results.get("win_rate", 0),
        )
        all_trades.extend(trades)

    # Build DataFrame (chronological — NO shuffling)
    df = pd.DataFrame(all_trades)
    df = df.sort_values("entry_date").reset_index(drop=True)

    # Save CSV
    csv_path = COMPASS_DIR / f"training_data_{config_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d trades to %s", len(df), csv_path)

    # Generate feature analysis
    analysis = generate_feature_analysis(df, config_name)
    analysis_path = COMPASS_DIR / f"feature_analysis_{config_name}.md"
    with open(analysis_path, "w") as f:
        f.write(analysis)
    logger.info("Saved feature analysis to %s", analysis_path)

    # Summary
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Total trades: %d", len(df))
    logger.info("Date range: %s to %s", df["entry_date"].min(), df["entry_date"].max())
    logger.info("Strategy split: %s", df["strategy_type"].value_counts().to_dict())
    logger.info("Overall win rate: %.1f%%", df["win"].mean() * 100)
    logger.info("Elapsed: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
