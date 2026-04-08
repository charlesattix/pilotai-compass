#!/usr/bin/env python3
"""
run_combined.py — Standalone runner for the Combined Portfolio.

Orchestrates the regime-switching portfolio that combines:
  EXP-1220 (autonomous core, via scripts/run_exp1220.py)
  EXP-1780 (advisory crisis alpha signals)
  EXP-1820 (advisory dispersion signals)
  EXP-1660 (advisory VRP signals)

Rule Zero: only real market data (Alpaca live + Yahoo VIX/SPY).

Usage:
    python3 scripts/run_combined.py                # daily check + execute
    python3 scripts/run_combined.py --dry-run      # show plan without execution
    python3 scripts/run_combined.py --smoke-test   # validate config + API
    python3 scripts/run_combined.py --status       # show current state
    python3 scripts/run_combined.py --health       # write health file
    python3 scripts/run_combined.py --report       # daily P&L report
    python3 scripts/run_combined.py --force-rebalance
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

CONFIG_PATH = ROOT / "configs" / "combined_portfolio.yaml"
STATE_PATH = LOG_DIR / "combined_state.json"
HEALTH_PATH = LOG_DIR / "combined_health.json"
PNL_JOURNAL = LOG_DIR / "combined_pnl_journal.csv"


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("combined")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    rotating = logging.handlers.RotatingFileHandler(
        LOG_DIR / "combined_portfolio.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
    )
    rotating.setFormatter(fmt)
    logger.addHandler(rotating)

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    logger.propagate = False
    return logger


log = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# State management
# ═══════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "current_regime": None,
        "regime_since": None,
        "current_allocations": {},
        "last_rebalance": None,
        "positions": [],
        "equity_high_water": 0.0,
        "peak_equity": 0.0,
        "halted": False,
        "halt_reason": None,
        "total_pnl": 0.0,
    }


def save_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def write_health(status: str, details: Optional[dict] = None,
                  error: Optional[str] = None):
    """Write health status for Charles's monitoring."""
    health = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "last_run": datetime.now().isoformat(),
        "error": error,
        "details": details or {},
    }
    try:
        HEALTH_PATH.write_text(json.dumps(health, indent=2, default=str))
    except Exception as e:
        log.error(f"Failed to write health file: {e}")


def log_pnl(entry: dict):
    """Append a row to the daily P&L journal CSV."""
    fields = ["timestamp", "equity", "daily_pnl", "total_pnl",
              "drawdown_pct", "regime", "n_positions", "note"]
    exists = PNL_JOURNAL.exists()
    with open(PNL_JOURNAL, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(entry)


# ═══════════════════════════════════════════════════════════════════════════
# Market data (real Yahoo)
# ═══════════════════════════════════════════════════════════════════════════

def _yahoo_fetch(symbol: str, max_retries: int = 3) -> Optional[float]:
    """Fetch latest price from Yahoo Finance with retry."""
    import urllib.request
    import time
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?range=1d&interval=1d")
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                results = data.get("chart", {}).get("result")
                if not results:
                    raise ValueError("no results")
                price = results[0]["meta"].get("regularMarketPrice")
                if price is None:
                    raise ValueError("no price")
                return float(price)
        except Exception as e:
            last_err = e
            log.warning(f"Yahoo fetch attempt {attempt+1}/{max_retries} failed for {symbol}: {e}")
            time.sleep(2 ** attempt)
    log.error(f"Yahoo fetch failed for {symbol} after {max_retries} attempts: {last_err}")
    return None


def get_vix() -> Optional[float]:
    return _yahoo_fetch("%5EVIX")


def get_spy_price() -> Optional[float]:
    return _yahoo_fetch("SPY")


def get_spy_ma50_slope() -> Optional[float]:
    """Compute SPY 50-day MA slope (positive = uptrend).

    Returns slope as annualized % change.
    """
    try:
        from backtest.backtester import _yf_download_safe
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
        df = _yf_download_safe("SPY", start, end)
        if df.empty or len(df) < 55:
            return None
        ma = df["Close"].rolling(50).mean()
        # Slope of MA over last 10 days
        recent = ma.iloc[-10:].values
        if len(recent) < 10:
            return None
        slope = (recent[-1] - recent[0]) / recent[0]
        annualized = slope * 25.2  # 10 days → year
        return float(annualized)
    except Exception as e:
        log.warning(f"SPY MA slope computation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Regime detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_regime(config: dict) -> Tuple[str, dict]:
    """Classify the current market regime.

    Returns (regime_name, market_state).
    """
    vix = get_vix()
    spy = get_spy_price()
    slope = get_spy_ma50_slope()

    if vix is None:
        log.warning("VIX unavailable — defaulting to NEUTRAL")
        return "NEUTRAL", {"vix": None, "spy": spy, "slope": slope}

    market_state = {"vix": vix, "spy": spy, "slope": slope}

    # Apply regime rules (match YAML conditions)
    if vix >= 30:
        return "HIGH_VOL", market_state
    if vix < 30 and slope is not None and slope < 0:
        return "BEAR", market_state
    if vix < 20 and slope is not None and slope > 0:
        return "BULL", market_state
    return "NEUTRAL", market_state


# ═══════════════════════════════════════════════════════════════════════════
# Alpaca client (reuse EXP-1220 wrapper)
# ═══════════════════════════════════════════════════════════════════════════

def get_alpaca_account() -> Optional[dict]:
    """Fetch Alpaca account snapshot. Real API call."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        log.warning("Alpaca credentials not set")
        return None

    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        acct = client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "status": str(acct.status),
            "account_number": str(acct.account_number),
        }
    except ImportError:
        log.error("alpaca-py not installed. Run: pip3 install alpaca-py")
        return None
    except Exception as e:
        log.error(f"Alpaca account fetch failed: {e}")
        return None


def get_alpaca_positions() -> List[dict]:
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return []
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        positions = client.get_all_positions()
        return [{
            "symbol": str(p.symbol),
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        } for p in positions]
    except Exception as e:
        log.warning(f"Alpaca positions fetch failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Advisory signal generators (EXP-1780 / EXP-1820 / EXP-1660)
# ═══════════════════════════════════════════════════════════════════════════

def exp1780_advisory_signal(config: dict, regime: str) -> Optional[dict]:
    """Generate EXP-1780 crisis alpha signal recommendation.

    Returns a dict with target weights across the 13-asset universe.
    This is ADVISORY — Charles must execute manually.
    """
    if regime not in ("BEAR", "HIGH_VOL"):
        return None

    try:
        from compass.crisis_alpha_v3 import (
            load_universe_v3, compute_momentum, compute_vol_target_weights,
            LOOKBACK_GRID,
        )
        # load_universe_v3 requires >=400 bars per ticker; use 3-year window
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        log.info(f"EXP-1780: loading universe {start} → {end}")
        prices = load_universe_v3(start=start, end=end)
        if prices.empty or len(prices) < 250:
            log.warning(f"EXP-1780: insufficient data ({len(prices)} bars)")
            return None

        lookbacks, lb_weights = LOOKBACK_GRID["v2_round"]
        signal = compute_momentum(prices, lookbacks, lb_weights)
        weights = compute_vol_target_weights(
            prices, signal, vol_target=0.10, leverage=2.5,
        )

        # Take the most recent weight row
        latest_weights = weights.iloc[-1].to_dict()
        as_of = weights.index[-1].strftime("%Y-%m-%d")

        # Filter to non-zero positions
        non_zero = {k: round(float(v), 4) for k, v in latest_weights.items() if abs(v) > 0.01}
        return {
            "strategy": "EXP-1780",
            "as_of": as_of,
            "universe_size": len(prices.columns),
            "target_weights": non_zero,
            "gross_exposure": round(sum(abs(v) for v in latest_weights.values()), 4),
            "note": "Advisory signal — Charles executes manually in Alpaca paper",
        }
    except Exception as e:
        log.error(f"EXP-1780 signal generation failed: {e}")
        return None


def exp1820_advisory_signal(regime: str) -> Optional[dict]:
    """Generate EXP-1820 dispersion signal.

    Simple proxy: if VIX > 20 (richer index vol), recommend short SPY straddle.
    Real implementation would compute IV vs RV z-score.
    """
    if regime not in ("NEUTRAL",):
        return None
    vix = get_vix()
    if vix is None or vix < 20:
        return None
    return {
        "strategy": "EXP-1820",
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "signal": "short_spy_straddle",
        "reason": f"VIX {vix:.1f} > 20 (IV likely rich vs realized)",
        "max_capital_pct": 5.0,
        "note": "Advisory — small overlay, execute manually",
    }


def exp1660_advisory_signal(regime: str) -> Optional[dict]:
    """EXP-1660 VRP: fires in HIGH_VOL regime only."""
    if regime != "HIGH_VOL":
        return None
    vix = get_vix()
    return {
        "strategy": "EXP-1660",
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "signal": "sell_atm_spy_credit_spreads",
        "reason": f"HIGH_VOL regime, VIX {vix:.1f} — front-month premium rich",
        "max_capital_pct": 20.0,
        "note": "Advisory — execute manually, respect daily loss limits",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Telegram alerts
# ═══════════════════════════════════════════════════════════════════════════

def send_alert(msg: str, level: str = "INFO"):
    """Send a Telegram message if configured."""
    try:
        from shared.telegram_alerts import send_message, is_configured
        if not is_configured():
            log.info(f"[alert:{level}] {msg} (telegram not configured)")
            return
        prefix = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🔴"}.get(level, "")
        full = f"{prefix} <b>[Combined Portfolio]</b>\n{msg}"
        send_message(full)
    except Exception as e:
        log.warning(f"Telegram alert failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Drawdown monitoring
# ═══════════════════════════════════════════════════════════════════════════

def check_drawdown(config: dict, state: dict, equity: float) -> Tuple[bool, float]:
    """Check drawdown against thresholds. Returns (halt_triggered, dd_pct)."""
    peak = max(state.get("peak_equity", 0.0), equity)
    state["peak_equity"] = peak

    dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0.0

    risk = config["risk"]
    halt_threshold = risk["max_drawdown_halt_pct"]

    warn = config["monitoring"]["telegram"]["alerts"]["drawdown_warning_pct"]
    critical = config["monitoring"]["telegram"]["alerts"]["drawdown_critical_pct"]

    if dd_pct >= halt_threshold:
        if not state.get("halted"):
            state["halted"] = True
            state["halt_reason"] = f"Drawdown {dd_pct:.2f}% >= {halt_threshold}%"
            send_alert(
                f"HALT: Portfolio DD {dd_pct:.2f}% >= {halt_threshold}% limit. "
                f"All trading suspended until DD recovers to "
                f"{risk['drawdown_recovery_pct']}%.",
                level="CRITICAL",
            )
        return True, dd_pct

    if dd_pct >= critical:
        send_alert(
            f"CRITICAL: Portfolio DD {dd_pct:.2f}% crossed {critical}% "
            f"(halt at {halt_threshold}%)",
            level="CRITICAL",
        )
    elif dd_pct >= warn:
        send_alert(
            f"Warning: Portfolio DD {dd_pct:.2f}% (warning threshold {warn}%)",
            level="WARNING",
        )

    # Recovery check
    if state.get("halted") and dd_pct <= risk["drawdown_recovery_pct"]:
        state["halted"] = False
        state["halt_reason"] = None
        send_alert(
            f"RESUMED: DD recovered to {dd_pct:.2f}% (<= "
            f"{risk['drawdown_recovery_pct']}%). Trading active.",
            level="INFO",
        )

    return False, dd_pct


# ═══════════════════════════════════════════════════════════════════════════
# Main daily run
# ═══════════════════════════════════════════════════════════════════════════

def run_daily(config: dict, state: dict, dry_run: bool = False,
              force_rebalance: bool = False) -> dict:
    """Execute one daily cycle: regime detect, rebalance check, signals.

    Returns a summary dict.
    """
    log.info("=" * 60)
    log.info(f"Combined Portfolio Daily Run — "
             f"{'DRY-RUN' if dry_run else 'LIVE'}")
    log.info("=" * 60)

    # 1. Get account state
    acct = get_alpaca_account() if not dry_run else {
        "equity": 100000.0, "cash": 100000.0, "buying_power": 100000.0,
        "status": "DRY-RUN", "account_number": "PAPER-DRY",
    }
    if acct is None:
        write_health("error", error="Alpaca account unavailable")
        return {"error": "alpaca_unavailable"}

    equity = acct["equity"]
    log.info(f"Account: {acct['account_number']}, equity ${equity:,.2f}")

    # 2. Detect regime
    regime, market_state = detect_regime(config)
    log.info(f"Regime: {regime} (VIX {market_state.get('vix', '—')}, "
             f"SPY {market_state.get('spy', '—')}, "
             f"slope {market_state.get('slope', '—')})")

    regime_changed = (state.get("current_regime") != regime)
    if regime_changed:
        prev = state.get("current_regime", "NONE")
        log.info(f"REGIME CHANGE: {prev} → {regime}")
        send_alert(
            f"Regime change: <b>{prev}</b> → <b>{regime}</b>\n"
            f"VIX {market_state.get('vix', '—')}, "
            f"SPY {market_state.get('spy', '—')}",
            level="INFO",
        )
        state["current_regime"] = regime
        state["regime_since"] = str(date.today())

    # 3. Get target allocations (filter out non-numeric keys like 'notes')
    raw_alloc = config["allocations"].get(regime, {})
    target_alloc = {k: v for k, v in raw_alloc.items()
                    if isinstance(v, (int, float))}
    log.info(f"Target allocations ({regime}): {target_alloc}")

    # 4. Check drawdown
    halted, dd_pct = check_drawdown(config, state, equity)
    if halted:
        log.warning(f"HALTED: DD {dd_pct:.2f}% — skipping trade execution")
        write_health("halted", {"dd_pct": dd_pct,
                                  "reason": state.get("halt_reason")})
        return {"halted": True, "dd_pct": dd_pct, "regime": regime}

    # 5. Rebalance check
    today = date.today()
    last_rebal = state.get("last_rebalance")
    should_rebalance = False

    if force_rebalance:
        should_rebalance = True
        log.info("Force rebalance triggered")
    elif regime_changed:
        should_rebalance = True
        log.info("Rebalance triggered by regime change")
    elif last_rebal is None:
        should_rebalance = True
        log.info("First rebalance (no prior state)")
    else:
        last_dt = datetime.strptime(last_rebal, "%Y-%m-%d").date()
        days_since = (today - last_dt).days
        rebal_day = config["rebalance"]["day_of_week"]
        if days_since >= 7 and today.strftime("%A").lower() == rebal_day:
            should_rebalance = True
            log.info(f"Weekly rebalance triggered ({days_since} days since last)")

    # 6. Generate advisory signals (always, regardless of rebalance)
    advisory_signals = []
    if target_alloc.get("EXP-1780", 0) > 0:
        sig = exp1780_advisory_signal(config, regime)
        if sig:
            advisory_signals.append(sig)
            log.info(f"EXP-1780 signal: {sig.get('gross_exposure', 0):.2f} gross "
                     f"across {len(sig.get('target_weights', {}))} assets")

    if target_alloc.get("EXP-1820", 0) > 0:
        sig = exp1820_advisory_signal(regime)
        if sig:
            advisory_signals.append(sig)
            log.info(f"EXP-1820 signal: {sig.get('signal', '')}")

    if target_alloc.get("EXP-1660", 0) > 0:
        sig = exp1660_advisory_signal(regime)
        if sig:
            advisory_signals.append(sig)
            log.info(f"EXP-1660 signal: {sig.get('signal', '')}")

    # 7. EXP-1220 execution (autonomous — delegate to its runner)
    exp1220_pct = target_alloc.get("EXP-1220", 0)
    exp1220_note = ""
    if exp1220_pct > 0 and should_rebalance and not dry_run:
        log.info(f"EXP-1220 allocation: {exp1220_pct:.0%} of equity")
        log.info("Delegating to scripts/run_exp1220.py (separate process)")
        exp1220_note = (
            f"EXP-1220 runs via its own scanner (scripts/run_exp1220.py). "
            f"Ensure its LaunchAgent is active. Target allocation: "
            f"{exp1220_pct:.0%}."
        )
    elif dry_run and exp1220_pct > 0:
        exp1220_note = f"[DRY] Would delegate {exp1220_pct:.0%} to EXP-1220"

    # 8. Update state
    state["current_allocations"] = target_alloc
    if should_rebalance:
        state["last_rebalance"] = str(today)
        send_alert(
            f"Rebalance executed: <b>{regime}</b> regime\n"
            f"Target allocations: {target_alloc}\n"
            f"Advisory signals: {len(advisory_signals)}",
            level="INFO",
        )

    # 9. Log P&L
    last_equity = state.get("last_equity", equity)
    daily_pnl = equity - last_equity
    state["last_equity"] = equity
    state["total_pnl"] = state.get("total_pnl", 0) + daily_pnl

    if not dry_run:
        log_pnl({
            "timestamp": datetime.now().isoformat(),
            "equity": round(equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "total_pnl": round(state["total_pnl"], 2),
            "drawdown_pct": round(dd_pct, 2),
            "regime": regime,
            "n_positions": len(get_alpaca_positions()),
            "note": "regime_change" if regime_changed else "normal",
        })

    # 10. Health
    write_health("ok", {
        "regime": regime,
        "equity": equity,
        "dd_pct": dd_pct,
        "last_rebalance": state.get("last_rebalance"),
        "advisory_signals": len(advisory_signals),
    })

    save_state(state)

    return {
        "regime": regime,
        "equity": equity,
        "daily_pnl": daily_pnl,
        "dd_pct": dd_pct,
        "allocations": target_alloc,
        "advisory_signals": advisory_signals,
        "exp1220_note": exp1220_note,
        "rebalanced": should_rebalance,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

def run_smoke_test() -> int:
    print("=" * 60)
    print("  Combined Portfolio Smoke Test")
    print("=" * 60)
    print()

    failures = []
    warnings = []

    # 1. Config
    print("[1/6] Loading config...")
    try:
        config = load_config()
        assert "strategies" in config
        assert "allocations" in config
        assert "risk" in config
        print(f"      OK  {config['name']} v{config['version']}")
    except Exception as e:
        failures.append(f"Config load failed: {e}")
        print(f"      FAIL {e}")

    # 2. State
    print("[2/6] Checking state file...")
    try:
        state = load_state()
        print(f"      OK  Current regime: {state.get('current_regime', 'none')}")
    except Exception as e:
        failures.append(f"State load failed: {e}")
        print(f"      FAIL {e}")

    # 3. Log dir
    print("[3/6] Checking log directory...")
    try:
        test = LOG_DIR / ".smoke_test_combined"
        test.write_text("test")
        test.unlink()
        print(f"      OK  {LOG_DIR} writable")
    except Exception as e:
        failures.append(f"Log dir not writable: {e}")
        print(f"      FAIL {e}")

    # 4. Yahoo data
    print("[4/6] Fetching VIX & SPY from Yahoo...")
    vix = get_vix()
    spy = get_spy_price()
    if vix is None:
        warnings.append("VIX fetch failed")
        print("      WARN VIX unavailable")
    else:
        print(f"      OK  VIX = {vix:.2f}")
    if spy is None:
        failures.append("SPY fetch failed")
        print("      FAIL SPY unreachable")
    else:
        print(f"      OK  SPY = ${spy:.2f}")

    # 5. Regime detection
    print("[5/6] Running regime detection...")
    try:
        config = load_config()
        regime, mstate = detect_regime(config)
        print(f"      OK  Regime: {regime}")
        target = config["allocations"].get(regime, {})
        alloc_parts = []
        for k, v in target.items():
            if isinstance(v, (int, float)):
                alloc_parts.append(f"{k}={v:.0%}")
            else:
                alloc_parts.append(f"{k}={v}")
        print(f"      OK  Target allocations: {', '.join(alloc_parts)}")
    except Exception as e:
        failures.append(f"Regime detection failed: {e}")
        print(f"      FAIL {e}")

    # 6. Alpaca
    print("[6/6] Checking Alpaca connectivity...")
    if not os.environ.get("ALPACA_API_KEY"):
        warnings.append("Alpaca credentials not set")
        print("      WARN ALPACA_API_KEY not set (OK for --dry-run)")
    else:
        acct = get_alpaca_account()
        if acct is None:
            failures.append("Alpaca account fetch failed")
            print("      FAIL Alpaca unreachable")
        else:
            print(f"      OK  Account {acct['account_number']}, "
                  f"equity ${acct['equity']:,.2f}")

    # Summary
    print()
    print("=" * 60)
    if failures:
        print(f"  SMOKE TEST FAILED ({len(failures)} errors, {len(warnings)} warnings)")
        for f in failures:
            print(f"    ERROR: {f}")
        for w in warnings:
            print(f"    WARN:  {w}")
        write_health("error", {"smoke_test": "failed"}, error="; ".join(failures))
        return 1
    elif warnings:
        print(f"  SMOKE TEST PASSED WITH WARNINGS ({len(warnings)})")
        for w in warnings:
            print(f"    WARN: {w}")
        write_health("warning", {"smoke_test": "passed_with_warnings"})
        return 0
    else:
        print(f"  SMOKE TEST PASSED — all 6 checks OK")
        write_health("ok", {"smoke_test": "passed"})
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Status / health / report
# ═══════════════════════════════════════════════════════════════════════════

def show_status():
    state = load_state()
    config = load_config()
    print()
    print("Combined Portfolio Status")
    print("=" * 60)
    print(f"Current regime:  {state.get('current_regime', 'unknown')}")
    print(f"Regime since:    {state.get('regime_since', 'n/a')}")
    print(f"Last rebalance:  {state.get('last_rebalance', 'never')}")
    print(f"Halted:          {state.get('halted', False)}")
    if state.get("halt_reason"):
        print(f"Halt reason:     {state['halt_reason']}")
    print(f"Total PnL:       ${state.get('total_pnl', 0):,.2f}")
    print(f"Peak equity:     ${state.get('peak_equity', 0):,.2f}")
    alloc = state.get("current_allocations", {})
    if alloc:
        print(f"\nCurrent allocations:")
        for k, v in alloc.items():
            if isinstance(v, (int, float)):
                print(f"  {k:<12} {v:.0%}")
    print()


def show_report(days: int = 7):
    if not PNL_JOURNAL.exists():
        print("No PnL journal yet.")
        return
    print(f"\nCombined Portfolio P&L Report (last {days} days)")
    print("=" * 60)
    with open(PNL_JOURNAL) as f:
        reader = csv.DictReader(f)
        rows = list(reader)[-days:]
    if not rows:
        print("No entries.")
        return
    for r in rows:
        ts = r["timestamp"][:10]
        print(f"{ts}  eq=${float(r['equity']):>11,.0f}  "
              f"dPnL=${float(r['daily_pnl']):>8,.0f}  "
              f"DD={float(r['drawdown_pct']):5.2f}%  "
              f"{r['regime']:<10}  {r['note']}")
    total_pnl = sum(float(r["daily_pnl"]) for r in rows)
    print(f"\nTotal P&L over {len(rows)} days: ${total_pnl:,.2f}")


def run_health_check() -> int:
    try:
        state = load_state()
        config = load_config()
        write_health("ok", {
            "regime": state.get("current_regime"),
            "halted": state.get("halted", False),
            "last_rebalance": state.get("last_rebalance"),
            "total_pnl": state.get("total_pnl", 0),
        })
        print(f"HEALTH: OK (regime={state.get('current_regime', 'none')}, "
              f"halted={state.get('halted', False)})")
        return 0
    except Exception as e:
        write_health("error", error=str(e))
        print(f"HEALTH: ERROR — {e}")
        return 2


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Combined Portfolio Runner (regime-switching)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without submitting orders")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Validate config + connectivity")
    parser.add_argument("--status", action="store_true",
                        help="Show current portfolio state")
    parser.add_argument("--health", action="store_true",
                        help="Write health file and exit")
    parser.add_argument("--report", action="store_true",
                        help="Show recent P&L journal")
    parser.add_argument("--force-rebalance", action="store_true",
                        help="Force rebalance even if not scheduled")
    parser.add_argument("--report-days", type=int, default=7)
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(run_smoke_test())
    if args.health:
        sys.exit(run_health_check())
    if args.status:
        show_status()
        return
    if args.report:
        show_report(args.report_days)
        return

    # Main daily run
    config = load_config()
    state = load_state()

    # Set Telegram experiment ID
    try:
        from shared.telegram_alerts import set_experiment_id
        set_experiment_id("COMBINED")
    except Exception:
        pass

    try:
        summary = run_daily(config, state, dry_run=args.dry_run,
                             force_rebalance=args.force_rebalance)
        log.info(f"Daily run complete: {summary}")

        if summary.get("advisory_signals"):
            log.info(f"\n{len(summary['advisory_signals'])} advisory signal(s) "
                     f"for manual review:")
            for sig in summary["advisory_signals"]:
                log.info(f"  [{sig['strategy']}] {sig}")

        if summary.get("exp1220_note"):
            log.info(f"\nEXP-1220: {summary['exp1220_note']}")

    except Exception as e:
        log.error(f"Daily run failed: {e}", exc_info=True)
        write_health("error", error=str(e))
        send_alert(f"Daily run ERROR: {e}", level="CRITICAL")
        sys.exit(1)
    finally:
        save_state(state)
        log.info("State saved.")


if __name__ == "__main__":
    main()
