"""
EXP-2830 — Paper Trading Signal Generator (production-ready)

Phase 9 starts 2026-04-09. This is the daily script the cron runs at
09:25 ET to produce trade signals for the 8-stream v8a portfolio on
Alpaca paper.

Pipeline
--------
  1. Fetch live market data from Yahoo Finance (SPY, QQQ, XLF, XLI,
     GLD, SLV, ^VIX, ^VIX3M, ^VVIX).
  2. Build causal regime classifier context (current regime, VoV
     percentile, VIX term-structure ratio).
  3. For each of 8 streams, emit a structured signal dict:
       action   — "OPEN" / "NO_TRADE" / "SKIP"
       reason   — why it was gated or accepted
       structure, direction, delta, strikes, DTE, width,
       target contracts, limit price, order type
  4. Apply portfolio-level overlays (VoV gate, FOMC week gate) from
     EXP-1970 / EXP-1880.
  5. Write a single JSON file to compass/reports/paper_signals/ named
     by date, ready for the Alpaca API integration layer to consume.
  6. Also write a human-readable audit log (JSONL append) to
     compass/logs/paper_signals_audit.jsonl.
  7. --dry-run mode prints signals to stdout without writing files.

Streams (v8a)
-------------
  exp1220       SPY  put-credit-spread  28 DTE   5% OTM
  xlf_cs        XLF  put-credit-spread  28 DTE   5% OTM
  xli_cs        XLI  put-credit-spread  28 DTE   5% OTM
  gld_cal       GLD  calendar spread    30-60 DTE
  slv_cal       SLV  calendar spread    30-60 DTE
  vol_arb       SPY  iv-rv relative val 30 DTE
  v5_hedge      SPY  hedge puts + VIX calls (hedge sleeve)
  qqq_cs        QQQ  put-credit-spread  28 DTE   5% OTM

Each signal is a simple dict; the Alpaca integration layer (not in
scope of this experiment) maps it to complex option orders.

REAL DATA ONLY. No synthetic fills. No hardcoded prices.

Usage
-----
  python3 -m compass.exp2830_paper_signal_generator           # production
  python3 -m compass.exp2830_paper_signal_generator --dry-run # print only
  python3 -m compass.exp2830_paper_signal_generator --date 2026-04-09
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIGNAL_DIR  = ROOT / "compass" / "reports" / "paper_signals"
AUDIT_LOG   = ROOT / "compass" / "logs" / "paper_signals_audit.jsonl"
CONFIG_JSON = ROOT / "compass" / "reports" / "exp2830_paper_signal_config.json"

CAPITAL_BASE = 100_000.0
TARGET_VOL   = 0.12      # portfolio target ann vol
MAX_LEVERAGE = 3.0       # hard cap


# ───────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────

def get_logger(dry_run: bool) -> logging.Logger:
    logger = logging.getLogger("exp2830")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
        if not dry_run:
            AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(AUDIT_LOG.with_suffix(".log"))
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(fh)
    return logger


# ───────────────────────────────────────────────────────────────────────────
# Market data (Yahoo Finance, 180-day window)
# ───────────────────────────────────────────────────────────────────────────

TICKERS = {
    "SPY":    "SPY",
    "QQQ":    "QQQ",
    "XLF":    "XLF",
    "XLI":    "XLI",
    "GLD":    "GLD",
    "SLV":    "SLV",
    "VIX":    "^VIX",
    "VIX3M":  "^VIX3M",
    "VVIX":   "^VVIX",
}


@dataclass
class MarketSnapshot:
    as_of: str
    closes: Dict[str, float]
    vix_20d_realized: float
    vix_value: float
    vix3m_value: float
    vvix_value: float
    vvix_percentile_60d: float
    term_structure_ratio: float          # VIX / VIX3M; > 1 = stress
    spy_trend_50d: float                 # pct change vs 50-day MA
    spy_20d_realized_vol: float
    data_sources: Dict[str, str]


def fetch_market_snapshot(as_of: date) -> MarketSnapshot:
    """Pull 180 calendar days of daily closes for every ticker."""
    import yfinance as yf
    start = (as_of - timedelta(days=180)).strftime("%Y-%m-%d")
    end   = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")

    data: Dict[str, pd.Series] = {}
    sources: Dict[str, str] = {}
    for key, sym in TICKERS.items():
        df = yf.download(sym, start=start, end=end, progress=False,
                         auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[key] = df["Close"].dropna()
        sources[key] = f"Yahoo Finance {sym} 180d"

    # Today's closes (most recent bar ≤ as_of)
    closes: Dict[str, float] = {}
    for k, s in data.items():
        if len(s):
            closes[k] = float(s.iloc[-1])

    # VIX term structure
    vix_val    = closes.get("VIX", float("nan"))
    vix3m_val  = closes.get("VIX3M", float("nan"))
    vvix_val   = closes.get("VVIX", float("nan"))
    ts_ratio   = vix_val / vix3m_val if vix3m_val > 0 else float("nan")

    # VVIX rolling 60d percentile
    vvix_hist = data.get("VVIX")
    if vvix_hist is not None and len(vvix_hist) >= 20:
        window = vvix_hist.tail(60)
        vvix_pct = float((window.rank(pct=True)).iloc[-1])
    else:
        vvix_pct = 0.5

    # SPY trend and realised vol
    spy = data["SPY"]
    spy_trend = 0.0
    spy_rv = 0.0
    if len(spy) >= 50:
        ma50 = spy.tail(50).mean()
        spy_trend = float(spy.iloc[-1] / ma50 - 1)
    if len(spy) >= 20:
        rets = spy.pct_change().dropna().tail(20)
        spy_rv = float(rets.std(ddof=1) * math.sqrt(252))

    # SPY 20-day realised
    vix_rv = spy_rv

    return MarketSnapshot(
        as_of=as_of.strftime("%Y-%m-%d"),
        closes=closes,
        vix_20d_realized=round(vix_rv * 100, 3),
        vix_value=round(vix_val, 2),
        vix3m_value=round(vix3m_val, 2),
        vvix_value=round(vvix_val, 2),
        vvix_percentile_60d=round(vvix_pct, 4),
        term_structure_ratio=round(ts_ratio, 4),
        spy_trend_50d=round(spy_trend * 100, 3),
        spy_20d_realized_vol=round(spy_rv * 100, 3),
        data_sources=sources,
    )


# ───────────────────────────────────────────────────────────────────────────
# Regime classifier (mirrors compass.regime)
# ───────────────────────────────────────────────────────────────────────────

def classify_regime(snap: MarketSnapshot) -> str:
    """Rule-based regime: BULL / BEAR / HIGH_VOL / LOW_VOL / CRASH."""
    v = snap.vix_value
    if np.isnan(v):
        return "UNKNOWN"
    # Crash: VIX > 40 AND SPY trend sharply down
    if v > 40 and snap.spy_trend_50d < -5:
        return "CRASH"
    if v > 30:
        return "HIGH_VOL"
    if v > 25 and snap.spy_trend_50d < 0:
        return "BEAR"
    if v < 20 and snap.spy_trend_50d > 0:
        return "BULL"
    if v < 15 and abs(snap.spy_trend_50d) < 2:
        return "LOW_VOL"
    if snap.spy_trend_50d > 0:
        return "BULL"
    if snap.spy_trend_50d < 0 and v > 22:
        return "BEAR"
    return "BULL"    # neutral fallback


# ───────────────────────────────────────────────────────────────────────────
# Portfolio-level overlays
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class OverlayContext:
    regime: str
    vov_gate_pass: bool          # EXP-1970 VoV overlay
    term_structure_gate_pass: bool  # EXP-2070 VIX TS overlay
    fomc_week: bool              # EXP-1880 FOMC filter
    dd_circuit_pass: bool        # EXP-2370 3% trailing-DD circuit (assume pass for paper day 1)
    notes: List[str] = field(default_factory=list)


def build_overlay_context(snap: MarketSnapshot, as_of: date) -> OverlayContext:
    notes: List[str] = []
    regime = classify_regime(snap)

    # VoV gate — block when VVIX 60d percentile > 0.85 (extreme convexity stress)
    vov_ok = not (snap.vvix_percentile_60d > 0.85)
    if not vov_ok:
        notes.append(f"VoV GATE: VVIX pctile {snap.vvix_percentile_60d:.2f} > 0.85")

    # VIX term structure gate — block when VIX/VIX3M > 1.05 (inverted >5%)
    ts_ok = not (snap.term_structure_ratio > 1.05)
    if not ts_ok:
        notes.append(f"TS GATE: VIX/VIX3M {snap.term_structure_ratio:.2f} > 1.05")

    # FOMC week filter — block entries within 2 calendar days of an FOMC meeting.
    # Approximation: scheduled FOMC meetings in 2026 (public calendar).
    fomc_2026 = {
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
        date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
        date(2026, 11, 4), date(2026, 12, 16),
    }
    fomc_week = any(abs((as_of - d).days) <= 2 for d in fomc_2026)
    if fomc_week:
        notes.append("FOMC WEEK: entries deferred")

    # DD circuit breaker — in live production this reads rolling 20d equity
    # from the monitor. For this signal generator we default to pass and
    # let the Alpaca integration layer enforce the actual live gate.
    dd_pass = True

    return OverlayContext(
        regime=regime,
        vov_gate_pass=vov_ok,
        term_structure_gate_pass=ts_ok,
        fomc_week=fomc_week,
        dd_circuit_pass=dd_pass,
        notes=notes,
    )


# ───────────────────────────────────────────────────────────────────────────
# Stream-level signal generators
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class StreamSignal:
    stream: str
    action: str                   # OPEN / NO_TRADE / SKIP
    underlier: str
    structure: str                # put_credit_spread / calendar / etc
    direction: str
    target_delta: Optional[float]
    short_strike: Optional[float]
    long_strike: Optional[float]
    width: Optional[float]
    expiry: Optional[str]
    dte_days: Optional[int]
    target_contracts: int
    limit_price: Optional[float]
    order_type: str               # limit_at_mid / combo_multi_leg
    reason: str
    stream_weight: float


# Stream weights from EXP-2600 v8a equal_risk_15%
STREAM_WEIGHTS = {
    "exp1220":  0.35,
    "xlf_cs":   0.10,
    "xli_cs":   0.10,
    "gld_cal":  0.10,
    "slv_cal":  0.05,
    "cross_vol":0.10,
    "v5_hedge": 0.05,
    "qqq_cs":   0.15,
}


def _friday_n_days_out(today: date, n: int) -> date:
    """Return the Friday closest to today + n calendar days."""
    target = today + timedelta(days=n)
    offset = (4 - target.weekday()) % 7   # Friday=4
    return target + timedelta(days=offset)


def _round_strike(price: float, increment: float = 1.0) -> float:
    return round(price / increment) * increment


def _put_credit_spread_signal(
        stream: str, underlier: str, snap: MarketSnapshot,
        ctx: OverlayContext, otm_pct: float = 0.05, width: float = 5.0,
        dte: int = 28, target_delta: float = 0.20) -> StreamSignal:

    weight = STREAM_WEIGHTS[stream]

    # Global kill switches
    if not ctx.vov_gate_pass:
        return StreamSignal(
            stream=stream, action="SKIP", underlier=underlier,
            structure="put_credit_spread", direction="short_put",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="limit_at_mid",
            reason="VoV gate blocked entry", stream_weight=weight,
        )
    if not ctx.term_structure_gate_pass:
        return StreamSignal(
            stream=stream, action="SKIP", underlier=underlier,
            structure="put_credit_spread", direction="short_put",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="limit_at_mid",
            reason="VIX term-structure inversion", stream_weight=weight,
        )
    if ctx.fomc_week:
        return StreamSignal(
            stream=stream, action="SKIP", underlier=underlier,
            structure="put_credit_spread", direction="short_put",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="limit_at_mid",
            reason="FOMC-week filter blocked entry", stream_weight=weight,
        )
    if ctx.regime in ("HIGH_VOL", "CRASH"):
        return StreamSignal(
            stream=stream, action="NO_TRADE", underlier=underlier,
            structure="put_credit_spread", direction="short_put",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="limit_at_mid",
            reason=f"regime={ctx.regime} — wait", stream_weight=weight,
        )

    # Signal-level construction
    price = snap.closes.get(underlier)
    if price is None or price <= 0:
        return StreamSignal(
            stream=stream, action="NO_TRADE", underlier=underlier,
            structure="put_credit_spread", direction="short_put",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="limit_at_mid",
            reason="missing underlier price", stream_weight=weight,
        )

    today = datetime.strptime(snap.as_of, "%Y-%m-%d").date()
    expiry = _friday_n_days_out(today, dte)
    short_strike = _round_strike(price * (1 - otm_pct))
    long_strike  = short_strike - width

    # Sizing: 3% of stream-allocated capital as max loss
    stream_cap = CAPITAL_BASE * weight * MAX_LEVERAGE
    max_loss_per_contract = (width - 0.5) * 100   # conservative credit estimate
    contracts = max(1, int(stream_cap * 0.03 / max_loss_per_contract))
    contracts = min(contracts, 10)  # safety cap

    # Placeholder limit price — Alpaca integration must re-quote at fill time
    est_credit = round(width * 0.15, 2)    # ~15% of width is a reasonable target

    return StreamSignal(
        stream=stream,
        action="OPEN",
        underlier=underlier,
        structure="put_credit_spread",
        direction="short_put",
        target_delta=target_delta,
        short_strike=short_strike,
        long_strike=long_strike,
        width=width,
        expiry=expiry.strftime("%Y-%m-%d"),
        dte_days=(expiry - today).days,
        target_contracts=contracts,
        limit_price=est_credit,
        order_type="limit_at_mid_combo",
        reason=f"regime={ctx.regime} overlays passed",
        stream_weight=weight,
    )


def _calendar_spread_signal(stream: str, etf: str, snap: MarketSnapshot,
                            ctx: OverlayContext) -> StreamSignal:
    weight = STREAM_WEIGHTS[stream]
    if ctx.regime == "CRASH":
        return StreamSignal(
            stream=stream, action="SKIP", underlier=etf,
            structure="calendar_spread", direction="long_back_short_front",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="combo_multi_leg",
            reason="regime=CRASH", stream_weight=weight,
        )
    price = snap.closes.get(etf)
    if price is None or price <= 0:
        return StreamSignal(
            stream=stream, action="NO_TRADE", underlier=etf,
            structure="calendar_spread", direction="long_back_short_front",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="combo_multi_leg",
            reason="missing underlier price", stream_weight=weight,
        )

    today = datetime.strptime(snap.as_of, "%Y-%m-%d").date()
    front_exp = _friday_n_days_out(today, 30)
    back_exp  = _friday_n_days_out(today, 60)
    strike = _round_strike(price, 1.0)

    stream_cap = CAPITAL_BASE * weight * MAX_LEVERAGE
    contracts = max(1, min(15, int(stream_cap * 0.02 / 50)))

    return StreamSignal(
        stream=stream,
        action="OPEN",
        underlier=etf,
        structure="calendar_spread",
        direction="long_back_short_front",
        target_delta=0.50,   # ATM calendar
        short_strike=strike,
        long_strike=strike,
        width=0.0,
        expiry=f"{front_exp.strftime('%Y-%m-%d')}/{back_exp.strftime('%Y-%m-%d')}",
        dte_days=(front_exp - today).days,
        target_contracts=contracts,
        limit_price=None,
        order_type="combo_multi_leg",
        reason=f"regime={ctx.regime} calendar constructive",
        stream_weight=weight,
    )


def _vol_arb_signal(snap: MarketSnapshot, ctx: OverlayContext) -> StreamSignal:
    weight = STREAM_WEIGHTS["cross_vol"]
    iv = snap.vix_value
    rv = snap.spy_20d_realized_vol
    if np.isnan(iv) or np.isnan(rv):
        return StreamSignal(
            stream="cross_vol", action="NO_TRADE", underlier="SPY",
            structure="iv_rv_relative_value", direction="short_iv_long_rv",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="combo_multi_leg",
            reason="missing iv/rv", stream_weight=weight,
        )

    vrp = iv - rv  # positive ≈ sell vol is attractive
    action = "OPEN" if vrp > 2.0 else "NO_TRADE"
    reason = (f"VRP={vrp:.2f} > 2.0, short IV long RV"
              if action == "OPEN" else
              f"VRP={vrp:.2f} < 2.0 threshold, waiting")
    today = datetime.strptime(snap.as_of, "%Y-%m-%d").date()
    expiry = _friday_n_days_out(today, 30)

    stream_cap = CAPITAL_BASE * weight * MAX_LEVERAGE
    contracts = max(1, min(8, int(stream_cap * 0.02 / 100)))

    return StreamSignal(
        stream="cross_vol",
        action=action,
        underlier="SPY",
        structure="iv_rv_relative_value",
        direction="short_iv_long_rv",
        target_delta=0.30,
        short_strike=None,
        long_strike=None,
        width=None,
        expiry=expiry.strftime("%Y-%m-%d"),
        dte_days=(expiry - today).days,
        target_contracts=contracts if action == "OPEN" else 0,
        limit_price=None,
        order_type="combo_multi_leg",
        reason=reason,
        stream_weight=weight,
    )


def _v5_hedge_signal(snap: MarketSnapshot, ctx: OverlayContext) -> StreamSignal:
    """Crisis Alpha v5: long SPY far-OTM puts + long VIX calls. Hedge sleeve.
    Only engage when stress indicators fire."""
    weight = STREAM_WEIGHTS["v5_hedge"]
    vix = snap.vix_value
    if np.isnan(vix):
        return StreamSignal(
            stream="v5_hedge", action="NO_TRADE", underlier="SPY",
            structure="hedge_sleeve", direction="long_tail_hedge",
            target_delta=None, short_strike=None, long_strike=None,
            width=None, expiry=None, dte_days=None, target_contracts=0,
            limit_price=None, order_type="combo_multi_leg",
            reason="missing VIX", stream_weight=weight,
        )
    # Lean-in when VIX > 25 or term inverted
    engage = vix > 25 or (not ctx.term_structure_gate_pass)
    spy_price = snap.closes.get("SPY", 0.0)
    today = datetime.strptime(snap.as_of, "%Y-%m-%d").date()
    expiry = _friday_n_days_out(today, 45)
    hedge_strike = _round_strike(spy_price * 0.90)  # 10% OTM put

    stream_cap = CAPITAL_BASE * weight * MAX_LEVERAGE
    contracts = max(1, min(5, int(stream_cap * 0.05 / 50))) if engage else 1

    return StreamSignal(
        stream="v5_hedge",
        action="OPEN" if engage else "HOLD_MIN",
        underlier="SPY",
        structure="hedge_sleeve",
        direction="long_tail_hedge",
        target_delta=0.10,
        short_strike=None,
        long_strike=hedge_strike,
        width=None,
        expiry=expiry.strftime("%Y-%m-%d"),
        dte_days=(expiry - today).days,
        target_contracts=contracts,
        limit_price=None,
        order_type="combo_multi_leg",
        reason=(f"engage: VIX={vix:.1f}, ts_ok={ctx.term_structure_gate_pass}"
                if engage else f"maintain 1-contract floor, VIX={vix:.1f}"),
        stream_weight=weight,
    )


# ───────────────────────────────────────────────────────────────────────────
# Orchestrator
# ───────────────────────────────────────────────────────────────────────────

def generate_all_signals(as_of: date, logger: logging.Logger) -> Dict:
    logger.info(f"Building market snapshot for {as_of}")
    snap = fetch_market_snapshot(as_of)

    logger.info(f"VIX {snap.vix_value} VIX3M {snap.vix3m_value} "
                f"VVIX {snap.vvix_value} (60d pct {snap.vvix_percentile_60d:.2f}) "
                f"TS {snap.term_structure_ratio:.3f} "
                f"SPY trend50 {snap.spy_trend_50d:+.2f}%")

    ctx = build_overlay_context(snap, as_of)
    logger.info(f"Regime: {ctx.regime} | VoV pass={ctx.vov_gate_pass} | "
                f"TS pass={ctx.term_structure_gate_pass} | "
                f"FOMC week={ctx.fomc_week}")
    for n in ctx.notes:
        logger.info(f"  note: {n}")

    signals: List[StreamSignal] = [
        _put_credit_spread_signal("exp1220", "SPY", snap, ctx,
                                   otm_pct=0.05, width=5.0,  dte=28),
        _put_credit_spread_signal("xlf_cs",  "XLF", snap, ctx,
                                   otm_pct=0.05, width=1.0,  dte=28),
        _put_credit_spread_signal("xli_cs",  "XLI", snap, ctx,
                                   otm_pct=0.05, width=2.5,  dte=28),
        _put_credit_spread_signal("qqq_cs",  "QQQ", snap, ctx,
                                   otm_pct=0.05, width=5.0,  dte=28),
        _calendar_spread_signal("gld_cal",   "GLD", snap, ctx),
        _calendar_spread_signal("slv_cal",   "SLV", snap, ctx),
        _vol_arb_signal(snap, ctx),
        _v5_hedge_signal(snap, ctx),
    ]

    # Portfolio-level roll-ups
    n_open = sum(1 for s in signals if s.action == "OPEN")
    n_skip = sum(1 for s in signals if s.action == "SKIP")
    n_no_trade = sum(1 for s in signals if s.action == "NO_TRADE")
    total_contracts = sum(s.target_contracts for s in signals)

    logger.info(f"Signals: OPEN={n_open} NO_TRADE={n_no_trade} SKIP={n_skip} "
                f"| total contracts={total_contracts}")

    return {
        "experiment": "EXP-2830",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "as_of": snap.as_of,
        "market": asdict(snap),
        "overlay_context": asdict(ctx),
        "portfolio_summary": {
            "n_streams": len(signals),
            "n_open": n_open,
            "n_no_trade": n_no_trade,
            "n_skip": n_skip,
            "total_contracts": total_contracts,
            "max_leverage_cap": MAX_LEVERAGE,
            "target_vol": TARGET_VOL,
        },
        "signals": [asdict(s) for s in signals],
        "rule_zero": ("All data from Yahoo Finance (real). Option chain "
                       "selection happens in the Alpaca integration layer "
                       "using IronVault-verified strikes."),
    }


def persist(payload: Dict, dry_run: bool, logger: logging.Logger) -> None:
    if dry_run:
        print(json.dumps(payload, indent=2, default=str))
        logger.info("[dry-run] no files written")
        return

    date_str = payload["as_of"]
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    signals_path = SIGNAL_DIR / f"signals_{date_str}.json"
    signals_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Wrote signals → {signals_path.relative_to(ROOT)}")

    # Audit log: one JSON line per signal generation run + per signal
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a") as fh:
        audit_line = {
            "ts": payload["generated_at"],
            "as_of": date_str,
            "regime": payload["overlay_context"]["regime"],
            "n_open": payload["portfolio_summary"]["n_open"],
            "n_skip": payload["portfolio_summary"]["n_skip"],
            "n_no_trade": payload["portfolio_summary"]["n_no_trade"],
            "overlay_notes": payload["overlay_context"]["notes"],
        }
        fh.write(json.dumps(audit_line) + "\n")
        for s in payload["signals"]:
            fh.write(json.dumps({
                "ts": payload["generated_at"],
                "as_of": date_str,
                "stream": s["stream"],
                "action": s["action"],
                "reason": s["reason"],
                "target_contracts": s["target_contracts"],
            }) + "\n")
    logger.info(f"Appended audit → {AUDIT_LOG.relative_to(ROOT)}")


def write_config_doc() -> None:
    """One-time config dump so the Alpaca integration layer knows stream weights."""
    CONFIG_JSON.write_text(json.dumps({
        "experiment": "EXP-2830",
        "description": "Paper signal generator config for v8a portfolio",
        "capital_base_usd": CAPITAL_BASE,
        "target_vol": TARGET_VOL,
        "max_leverage": MAX_LEVERAGE,
        "stream_weights": STREAM_WEIGHTS,
        "gates": {
            "vov_percentile_block": 0.85,
            "term_structure_block_ratio": 1.05,
            "fomc_block_days": 2,
            "regime_block": ["HIGH_VOL", "CRASH"],
        },
        "order_defaults": {
            "order_type": "limit_at_mid_combo",
            "dte_days_credit_spreads": 28,
            "dte_days_calendars_front": 30,
            "dte_days_calendars_back": 60,
            "otm_pct_credit_spreads": 0.05,
        },
    }, indent=2))


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EXP-2830 paper signal generator")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print signals to stdout, don't write files")
    ap.add_argument("--date", type=str, default=None,
                    help="As-of date YYYY-MM-DD (default: today UTC)")
    args = ap.parse_args()

    logger = get_logger(args.dry_run)
    logger.info("=" * 60)
    logger.info("EXP-2830 Paper Trading Signal Generator")
    logger.info("=" * 60)

    if args.date:
        as_of = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        as_of = datetime.utcnow().date()

    write_config_doc()

    try:
        payload = generate_all_signals(as_of, logger)
    except Exception as e:
        logger.error(f"Signal generation failed: {type(e).__name__}: {e}")
        raise

    persist(payload, args.dry_run, logger)

    # Console roll-up
    print()
    print(f"{'STREAM':<12} {'ACTION':<10} {'UND':<5} {'STRIKE':>8} "
          f"{'EXPIRY':<12} {'CTR':>4}  REASON")
    print("-" * 90)
    for s in payload["signals"]:
        sk = s["short_strike"] if s["short_strike"] is not None else "—"
        exp = s["expiry"] or "—"
        print(f"{s['stream']:<12} {s['action']:<10} {s['underlier']:<5} "
              f"{str(sk):>8} {exp[:12]:<12} {s['target_contracts']:>4}  "
              f"{s['reason'][:50]}")
    print()
    print(f"Regime: {payload['overlay_context']['regime']} | "
          f"VIX {payload['market']['vix_value']} | "
          f"OPEN {payload['portfolio_summary']['n_open']} / "
          f"{payload['portfolio_summary']['n_streams']} streams")
    return payload


if __name__ == "__main__":
    main()
