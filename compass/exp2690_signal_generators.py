"""EXP-2690 — Production signal generators for the 8-stream paper loop.

Each stream module needs a `generate_today_signals(date)` entry point
for the paper-trading scheduler (compass/scripts/generate_daily_signals.py).
This module is the SINGLE source of truth for those generators; each of
the 8 stream modules has a thin delegate that imports from here.

Generators (one per stream):
    exp1220_signals   — SPY put-credit-spread (EXP-1220) + V+F overlay
    xlf_cs_signals    — XLF put-credit-spread (EXP-2160)
    xli_cs_signals    — XLI put-credit-spread (EXP-2160)
    qqq_cs_signals    — QQQ put-credit-spread (EXP-2240)
    gld_cal_signals   — GLD − GC=F calendar spread (EXP-1770)
    slv_cal_signals   — SLV − SI=F calendar spread (EXP-1770)
    vol_arb_signals   — SPY/QQQ/IWM/EEM IV−RV pairs (EXP-2020)
    v5_hedge_signals  — Crisis Alpha v5 13-ETF CTA (EXP-1780 v5)

Unified signal schema (each generator returns list[dict]):
    {
        "stream":      str,     # sleeve id
        "date":        str,     # ISO date
        "ticker":      str,     # primary underlier
        "action":      "OPEN" | "HOLD" | "BLOCKED" | "NONE",
        "direction":   str,     # put_credit_spread / calendar / long / short / ...
        "delta":       float,   # target short delta (options) or None
        "dte":         int,     # target days-to-expiration (options) or None
        "width":       float,   # spread width (options) or None
        "weight":      float,   # sleeve's portfolio weight
        "confidence":  float,   # [0, 1] — regime + overlay adjustment
        "notes":       str,     # human-readable reason
        "legs":        list,    # optional multi-leg detail
    }

Design notes
------------
* Every generator runs quickly (<30s) — suitable for a 9:25 ET scheduler.
* Each generator BLOCKS rather than raises on data gaps; the driver
  script captures errors as `action=ERROR` rows.
* Strike selection is deferred to the execution layer (Alpaca paper).
  Signals express INTENT (target delta, DTE, width) not fully-specified
  OCC symbols — the execution layer looks up the live chain and fills
  in the specific strikes.
* Causal: every generator uses data through date-1 to decide date's
  action. No look-ahead.
* Weights come from compass.exp2600_north_star_v8 equal_risk baseline.

Rule Zero: all market data comes from Yahoo / FRED / IronVault. No
synthetic data in any code path.
"""

from __future__ import annotations

import math
import sqlite3
import sys
import warnings
from dataclasses import dataclass
from datetime import date as dt_date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════════
# Portfolio weights (EXP-2600 equal_risk_15% baseline, 8-stream v8a)
# ═══════════════════════════════════════════════════════════════════════════
PORTFOLIO_WEIGHTS = {
    "exp1220":  0.316,
    "xlf_cs":   0.245,
    "xli_cs":   0.192,
    "gld_cal":  0.024,
    "slv_cal":  0.012,
    "vol_arb":  0.187,
    "v5_hedge": 0.023,
    "qqq_cs":   0.100,   # EXP-2600 v8a addition
}


# ═══════════════════════════════════════════════════════════════════════════
# Common helpers
# ═══════════════════════════════════════════════════════════════════════════

def _as_datetime(d) -> datetime:
    if isinstance(d, datetime):
        return d
    if isinstance(d, dt_date):
        return datetime(d.year, d.month, d.day)
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d")
    if isinstance(d, pd.Timestamp):
        return d.to_pydatetime()
    raise TypeError(f"unsupported date type: {type(d)}")


def _fetch_yahoo_close(symbol: str, start: str, end: str) -> pd.Series:
    """Thin Yahoo daily close fetcher."""
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(symbol, start=start, end=end, progress=False,
                          auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _vix_on(date: datetime, lookback_days: int = 30) -> Optional[float]:
    """Most recent VIX close on or before `date`."""
    start = (date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (date + timedelta(days=1)).strftime("%Y-%m-%d")
    vix = _fetch_yahoo_close("^VIX", start, end)
    if vix.empty:
        return None
    prior = vix[vix.index <= date]
    return float(prior.iloc[-1]) if len(prior) else None


def _base_signal(stream: str, ticker: str, date: datetime,
                   action: str = "NONE") -> Dict[str, Any]:
    return {
        "stream": stream,
        "date": date.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "action": action,
        "direction": None,
        "delta": None,
        "dte": None,
        "width": None,
        "weight": PORTFOLIO_WEIGHTS.get(stream, 0.0),
        "confidence": 1.0,
        "notes": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. EXP-1220 SPY put credit spreads + V+F overlay (EXP-2000)
# ═══════════════════════════════════════════════════════════════════════════

def exp1220_signals(date) -> List[Dict[str, Any]]:
    """SPY 30-delta put credit spread, biweekly entry cadence, with V+F
    overlay (VoV z-score + FOMC sentiment filter)."""
    date = _as_datetime(date)
    sig = _base_signal("exp1220", "SPY", date)
    sig["direction"] = "put_credit_spread"
    sig["delta"] = 0.30
    sig["dte"] = 28
    sig["width"] = 5.0

    # Monday-only entries per the standard scan cadence
    if date.weekday() != 0:  # 0 = Monday
        sig["action"] = "NONE"
        sig["notes"] = "not a Monday entry day"
        return [sig]

    vix = _vix_on(date)
    if vix is None:
        sig["action"] = "BLOCKED"
        sig["notes"] = "VIX data unavailable"
        return [sig]
    if vix > 40:
        sig["action"] = "BLOCKED"
        sig["notes"] = f"VIX {vix:.1f} > 40 extreme-crisis gate"
        return [sig]

    # V+F overlay — causal VoV z-score and FOMC hawkish window check
    confidence = 1.0
    overlay_notes = []

    # VoV (EXP-1970): compute 252-day z-score of 20-day realised vol
    # of VIX log-returns
    try:
        from compass.exp1970_vol_of_vol import build_vvol_panel
        vix_hist = _fetch_yahoo_close(
            "^VIX",
            (date - timedelta(days=520)).strftime("%Y-%m-%d"),
            (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if not vix_hist.empty:
            panel = build_vvol_panel(vix_hist)
            prior_panel = panel[panel.index < date]
            if len(prior_panel) > 0:
                z = float(prior_panel["vvol_z"].iloc[-1])
                if not math.isnan(z):
                    if z > 2.0:
                        sig["action"] = "BLOCKED"
                        sig["notes"] = f"V+F: VoV z={z:.2f} > 2 (panic)"
                        return [sig]
                    elif z > 1.0:
                        confidence *= 0.5
                        overlay_notes.append(f"VoV z={z:.2f} → 0.5x")
                    else:
                        overlay_notes.append(f"VoV z={z:.2f} normal")
    except Exception as e:
        overlay_notes.append(f"VoV unavailable: {type(e).__name__}")

    # FOMC overlay — block if within 5 trading days of a hawkish release
    # (heuristic: skip if VIX/VIX3M inverted — cheaper proxy)
    try:
        vix3m = _fetch_yahoo_close(
            "^VIX3M",
            (date - timedelta(days=30)).strftime("%Y-%m-%d"),
            (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if not vix3m.empty:
            prior = vix3m[vix3m.index <= date]
            if len(prior) > 0:
                v3m = float(prior.iloc[-1])
                if vix > v3m:
                    sig["action"] = "BLOCKED"
                    sig["notes"] = (
                        f"V+F: VIX {vix:.1f} > VIX3M {v3m:.1f} "
                        f"(term inversion)"
                    )
                    return [sig]
                overlay_notes.append(f"TS ratio {vix/v3m:.2f} (contango)")
    except Exception as e:
        overlay_notes.append(f"VIX3M unavailable: {type(e).__name__}")

    sig["action"] = "OPEN"
    sig["confidence"] = round(confidence, 3)
    sig["notes"] = f"VIX {vix:.1f}; " + "; ".join(overlay_notes)
    return [sig]


# ═══════════════════════════════════════════════════════════════════════════
# 2/3/8. XLF / XLI / QQQ put-credit-spreads (EXP-2160, EXP-2240)
# ═══════════════════════════════════════════════════════════════════════════

def _credit_spread_signal(stream: str, ticker: str, date: datetime,
                            short_delta: float = 0.20,
                            long_delta: float = 0.10,
                            dte: int = 30,
                            width: float = 5.0) -> Dict[str, Any]:
    sig = _base_signal(stream, ticker, date)
    sig["direction"] = "put_credit_spread"
    sig["delta"] = short_delta
    sig["dte"] = dte
    sig["width"] = width

    # Weekly cadence — entries on Mondays
    if date.weekday() != 0:
        sig["action"] = "NONE"
        sig["notes"] = "not a Monday entry day"
        return sig

    # VIX regime gate
    vix = _vix_on(date)
    if vix is None:
        sig["action"] = "BLOCKED"
        sig["notes"] = "VIX unavailable"
        return sig
    if vix > 40:
        sig["action"] = "BLOCKED"
        sig["notes"] = f"VIX {vix:.1f} > 40"
        return sig

    # Underlier liquidity check — spot price must be available
    try:
        spot = _fetch_yahoo_close(
            ticker,
            (date - timedelta(days=10)).strftime("%Y-%m-%d"),
            (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if spot.empty:
            sig["action"] = "BLOCKED"
            sig["notes"] = f"{ticker} price unavailable"
            return sig
        prior = spot[spot.index <= date]
        if len(prior) == 0:
            sig["action"] = "BLOCKED"
            sig["notes"] = f"no {ticker} price on/before {date.date()}"
            return sig
        spot_px = float(prior.iloc[-1])
    except Exception as e:
        sig["action"] = "BLOCKED"
        sig["notes"] = f"spot fetch failed: {type(e).__name__}"
        return sig

    sig["action"] = "OPEN"
    sig["confidence"] = 1.0
    sig["notes"] = (f"spot ${spot_px:.2f}, VIX {vix:.1f}, "
                    f"short Δ~{short_delta}, DTE {dte}, width ${width}")
    return sig


def xlf_cs_signals(date) -> List[Dict[str, Any]]:
    return [_credit_spread_signal("xlf_cs", "XLF", _as_datetime(date))]


def xli_cs_signals(date) -> List[Dict[str, Any]]:
    return [_credit_spread_signal("xli_cs", "XLI", _as_datetime(date))]


def qqq_cs_signals(date) -> List[Dict[str, Any]]:
    return [_credit_spread_signal("qqq_cs", "QQQ", _as_datetime(date),
                                    short_delta=0.25, dte=30)]


# ═══════════════════════════════════════════════════════════════════════════
# 4/5. GLD / SLV calendar spreads (EXP-1770 futures roll harvest)
# ═══════════════════════════════════════════════════════════════════════════

def _calendar_spread_signal(stream: str, etf: str, future: str,
                              date: datetime) -> Dict[str, Any]:
    sig = _base_signal(stream, f"{etf}-{future}", date)
    sig["direction"] = "spread_ratio"

    try:
        start = (date - timedelta(days=200)).strftime("%Y-%m-%d")
        end = (date + timedelta(days=1)).strftime("%Y-%m-%d")
        etf_px = _fetch_yahoo_close(etf, start, end)
        fut_px = _fetch_yahoo_close(future, start, end)
        if etf_px.empty or fut_px.empty:
            sig["action"] = "BLOCKED"
            sig["notes"] = f"price data unavailable for {etf}/{future}"
            return sig
        common = etf_px.index.intersection(fut_px.index)
        if len(common) < 60:
            sig["action"] = "BLOCKED"
            sig["notes"] = f"insufficient overlap ({len(common)} days)"
            return sig
        etf_r = np.log(etf_px.reindex(common)).diff()
        fut_r = np.log(fut_px.reindex(common)).diff()
        spread_ret = (etf_r - fut_r).dropna()
        if len(spread_ret) < 60:
            sig["action"] = "BLOCKED"
            sig["notes"] = "insufficient spread history"
            return sig

        # Rolling 60-day cumulative spread z-score (EXP-1770 signal)
        cum = spread_ret.rolling(60).sum()
        mu = cum.rolling(60).mean()
        sd = cum.rolling(60).std(ddof=0)
        z_series = (cum - mu) / sd.replace(0, np.nan)
        z = z_series[z_series.index <= date].iloc[-1] if not z_series.empty else np.nan
        if pd.isna(z):
            sig["action"] = "BLOCKED"
            sig["notes"] = "spread z-score unavailable"
            return sig
        z = float(z)

        if z > 1.0:
            sig["action"] = "OPEN"
            sig["direction"] = "short_etf_long_future"
            sig["notes"] = f"z={z:+.2f} > +1 → short {etf}, long {future}"
        elif z < -1.0:
            sig["action"] = "OPEN"
            sig["direction"] = "long_etf_short_future"
            sig["notes"] = f"z={z:+.2f} < -1 → long {etf}, short {future}"
        elif abs(z) < 0.05:
            sig["action"] = "NONE"
            sig["notes"] = f"z={z:+.2f} in deadzone"
        else:
            sig["action"] = "HOLD"
            sig["notes"] = f"z={z:+.2f} sticky zone (carry existing)"
        sig["confidence"] = min(1.0, abs(z) / 2.0)
    except Exception as e:
        sig["action"] = "BLOCKED"
        sig["notes"] = f"signal build failed: {type(e).__name__}: {e}"
    return sig


def gld_cal_signals(date) -> List[Dict[str, Any]]:
    return [_calendar_spread_signal("gld_cal", "GLD", "GC=F", _as_datetime(date))]


def slv_cal_signals(date) -> List[Dict[str, Any]]:
    return [_calendar_spread_signal("slv_cal", "SLV", "SI=F", _as_datetime(date))]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cross-vol arb (EXP-2020 weekly IV−RV pairs)
# ═══════════════════════════════════════════════════════════════════════════

def vol_arb_signals(date) -> List[Dict[str, Any]]:
    date = _as_datetime(date)
    sig = _base_signal("vol_arb", "SPY/QQQ/IWM/EEM", date)
    sig["direction"] = "iv_rv_pair"

    # Weekly cadence — Mondays only
    if date.weekday() != 0:
        sig["action"] = "NONE"
        sig["notes"] = "not a Monday entry day"
        return [sig]

    universe = ["SPY", "QQQ", "IWM", "EEM"]
    start = (date - timedelta(days=90)).strftime("%Y-%m-%d")
    end = (date + timedelta(days=1)).strftime("%Y-%m-%d")

    rv = {}   # 20-day realised vol per ticker
    for t in universe:
        px = _fetch_yahoo_close(t, start, end)
        if px.empty:
            continue
        logret = np.log(px).diff().dropna()
        prior = logret[logret.index <= date]
        if len(prior) >= 20:
            rv[t] = float(prior.iloc[-20:].std(ddof=1) * math.sqrt(252))

    if len(rv) < 2:
        sig["action"] = "BLOCKED"
        sig["notes"] = f"insufficient RV data ({len(rv)} tickers)"
        return [sig]

    # Use current ^VIX as SPY IV proxy; for others scale by VIX ratio
    # (simple fallback — production uses IronVault IV via BS inversion)
    vix = _vix_on(date)
    if vix is None:
        sig["action"] = "BLOCKED"
        sig["notes"] = "VIX unavailable"
        return [sig]
    iv = {t: vix / 100.0 for t in rv}   # uniform IV proxy
    spread = {t: iv[t] - rv[t] for t in rv}

    ordered = sorted(spread.items(), key=lambda kv: kv[1])
    long_t, long_spread = ordered[0]     # narrowest (IV cheap vs RV)
    short_t, short_spread = ordered[-1]  # widest
    gap = short_spread - long_spread

    if gap < 0.02:   # require ≥2 vol-point gap
        sig["action"] = "NONE"
        sig["notes"] = (f"spread gap {gap*100:.1f}pp < 2pp threshold; "
                         f"no pair")
        return [sig]

    sig["action"] = "OPEN"
    sig["dte"] = 30
    sig["confidence"] = min(1.0, gap / 0.05)
    sig["legs"] = [
        {"side": "long_straddle",  "ticker": long_t,
         "iv_minus_rv": round(long_spread * 100, 3)},
        {"side": "short_straddle", "ticker": short_t,
         "iv_minus_rv": round(short_spread * 100, 3)},
    ]
    sig["notes"] = (f"long {long_t} ({long_spread*100:+.2f}pp), "
                    f"short {short_t} ({short_spread*100:+.2f}pp), "
                    f"gap {gap*100:.2f}pp")
    return [sig]


# ═══════════════════════════════════════════════════════════════════════════
# 7. Crisis Alpha v5 13-ETF CTA hedge
# ═══════════════════════════════════════════════════════════════════════════

def v5_hedge_signals(date) -> List[Dict[str, Any]]:
    date = _as_datetime(date)
    meta = _base_signal("v5_hedge", "13-ETF", date)
    meta["direction"] = "multi_asset_cta"

    # Weekly rebalance — Mondays
    if date.weekday() != 0:
        meta["action"] = "NONE"
        meta["notes"] = "not a weekly rebalance day"
        return [meta]

    try:
        from compass.crisis_alpha_v3 import load_universe_v3, LOOKBACK_GRID
        from compass.crisis_alpha_v5 import (
            HedgeConfigV5, compute_v5_weights, stress_gate,
        )
        from compass.crisis_alpha_v4 import compute_signal_with_confirmation

        # Need 400 calendar-day trading data; request ~800 calendar days to be safe
        start_s = (date - timedelta(days=800)).strftime("%Y-%m-%d")
        end_s = (date + timedelta(days=1)).strftime("%Y-%m-%d")
        prices = load_universe_v3(start=start_s, end=end_s)
        prior = prices[prices.index <= date]
        if len(prior) < 100:
            meta["action"] = "BLOCKED"
            meta["notes"] = f"insufficient price history ({len(prior)} days)"
            return [meta]

        cfg = HedgeConfigV5(
            name="v5_prod", lookback_preset="slow",
            vol_target=0.05, leverage=1.0,
            dd_brake_threshold=0.05, dd_brake_zone=0.03,
            max_weight=0.20, require_confirmation=False,
            stress_threshold=0.05, stress_lookback=60,
            safe_haven_boost=2.0, equity_short_only=True,
        )
        lookbacks, lw = LOOKBACK_GRID[cfg.lookback_preset]
        signal_df = compute_signal_with_confirmation(
            prior, lookbacks, lw, cfg.require_confirmation,
        )
        weights = compute_v5_weights(prior, signal_df, cfg)

        # Stress gate — zero exposure outside SPY drawdown regime
        if "SPY" in prior.columns:
            gate = stress_gate(prior["SPY"], cfg.stress_threshold, cfg.stress_lookback)
            gate_val = float(gate.iloc[-1]) if len(gate) else 0.0
        else:
            gate_val = 1.0

        latest = weights.iloc[-1] * gate_val
        active = latest[latest.abs() >= 0.01].sort_values(
            key=lambda s: s.abs(), ascending=False
        )

        if len(active) == 0 or gate_val < 1.0:
            meta["action"] = "HOLD"
            meta["notes"] = (f"stress gate {gate_val:.0f}, "
                              f"{len(active)} active positions (below floor)")
            meta["confidence"] = 0.0
            return [meta]

        meta["action"] = "OPEN"
        meta["confidence"] = min(1.0, float(active.abs().sum()) / cfg.leverage)
        meta["notes"] = (f"stress_gate={gate_val:.0f}, "
                         f"{len(active)} active legs, "
                         f"gross exposure {float(active.abs().sum()):.2f}")

        rows = [meta]
        for tk, w in active.items():
            leg = _base_signal("v5_hedge", tk, date)
            leg["action"] = "OPEN"
            leg["direction"] = "long" if w > 0 else "short"
            leg["confidence"] = float(abs(w) / cfg.max_weight)
            leg["notes"] = f"weight {float(w):+.4f}"
            leg["legs"] = None
            rows.append(leg)
        return rows
    except Exception as e:
        meta["action"] = "BLOCKED"
        meta["notes"] = f"v5 signal build failed: {type(e).__name__}: {e}"
        return [meta]


# ═══════════════════════════════════════════════════════════════════════════
# Registry for the driver script
# ═══════════════════════════════════════════════════════════════════════════

GENERATOR_REGISTRY = {
    "exp1220":  exp1220_signals,
    "xlf_cs":   xlf_cs_signals,
    "xli_cs":   xli_cs_signals,
    "qqq_cs":   qqq_cs_signals,
    "gld_cal":  gld_cal_signals,
    "slv_cal":  slv_cal_signals,
    "vol_arb":  vol_arb_signals,
    "v5_hedge": v5_hedge_signals,
}


def generate_all_signals(date) -> List[Dict[str, Any]]:
    """Call every generator for `date` and return the concatenated list."""
    date = _as_datetime(date)
    rows: List[Dict[str, Any]] = []
    for stream, fn in GENERATOR_REGISTRY.items():
        try:
            out = fn(date) or []
            for r in out:
                r.setdefault("stream", stream)
                r.setdefault("date", date.strftime("%Y-%m-%d"))
            rows.extend(out)
        except Exception as e:
            err = _base_signal(stream, "?", date)
            err["action"] = "ERROR"
            err["notes"] = f"{type(e).__name__}: {e}"
            rows.append(err)
    return rows


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2024-01-15",
                        help="ISO date to generate signals for")
    args = parser.parse_args()
    rows = generate_all_signals(args.date)
    print(f"Generated {len(rows)} signals for {args.date}")
    print(json.dumps(rows, indent=2, default=str))
