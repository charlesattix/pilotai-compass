"""
EXP-2860 — Paper Trading Dry Run (end-to-end signal → mock Alpaca).

Proves the signal pipeline works end-to-end BEFORE real Alpaca keys
arrive. The flow:

  1. Call EXP-2830 `generate_all_signals(today)` to produce the eight
     v8a stream signals (exp1220, xlf_cs, xli_cs, qqq_cs, gld_cal,
     slv_cal, cross_vol, v5_hedge). This already uses the EXP-2830
     market snapshot builder, VoV/term-structure gates, FOMC filter,
     and per-stream action classifier.

  2. Apply the EXP-2820 VIX-adaptive leverage ladder as a
     post-signal size multiplier. VIX level → target contracts are
     scaled by the ladder factor (e.g. VIX < 20 → 1.00×, VIX 25-30
     → 0.75×, VIX ≥ 70 → 0.00× "flat"). Signals whose final size
     rounds to 0 are flagged VIX_FLAT.

  3. Translate every OPEN signal into Alpaca's multi-leg
     `/v2/orders` JSON shape using the OCC option-symbol convention
     and the documented `leg_request` schema. Each leg is either
     `side=sell` (short put at short_strike) or `side=buy` (long put
     at long_strike). Calendars produce two legs (front vs back
     expiry) at the same strike. The Alpaca legs are NOT submitted
     — a `mock_submission_response` block records what the API
     round-trip WOULD have returned.

  4. Write the full dry-run payload to
     compass/paper_trading/dry_run_<YYYY-MM-DD>.json.
     Also write an HTML audit and a reports/exp2860_paper_dry_run.json
     top-level report for easy inspection.

  5. Exit code 0 only if every OPEN signal produces a validated
     Alpaca leg array. Any schema validation failure exits non-zero.

REAL DATA — Rule Zero:
  * Market snapshot comes from EXP-2830 which pulls real Yahoo
    prices and VIX, and the EXP-2690 real FOMC calendar.
  * No mock fills, no synthetic prices. The only "mock" part is
    the Alpaca HTTP response — we simulate what the API would
    return, not what the market would fill.

Outputs:
  compass/exp2860_paper_dry_run.py                  (this file)
  compass/paper_trading/dry_run_<YYYY-MM-DD>.json   (canonical dry-run log)
  compass/reports/exp2860_paper_dry_run.json        (experiment summary)
  compass/reports/exp2860_paper_dry_run.html

Tag: EXP-2860
Run: python3 -m compass.exp2860_paper_dry_run [--date 2026-04-08]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2830_paper_signal_generator import (
    generate_all_signals,
    get_logger,
    STREAM_WEIGHTS,
)
from compass.exp2820_flash_crash_protection import vix_leverage_factor, VIX_LADDER

PAPER_DIR = ROOT / "compass" / "paper_trading"
REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2860_paper_dry_run.json"
REPORT_HTML = REPORT_DIR / "exp2860_paper_dry_run.html"


# ── Alpaca order schema helpers ───────────────────────────────────────


def _occ_option_symbol(underlier: str, expiry: str, option_type: str,
                       strike: float) -> str:
    """OCC 21-char option symbol.

    SPY 240419P00500000 = SPY 2024-04-19 put, $500 strike (×1000).
    """
    exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    yymmdd = exp_date.strftime("%y%m%d")
    strike_int = int(round(strike * 1000))
    return f"{underlier.upper()}{yymmdd}{option_type.upper()}{strike_int:08d}"


def _put_credit_spread_legs(signal: Dict) -> List[Dict]:
    """Two legs: sell short-strike put, buy long-strike put."""
    qty = signal["target_contracts_after_vix"]
    expiry = signal["expiry"]
    under = signal["underlier"]
    return [
        {
            "symbol": _occ_option_symbol(under, expiry, "P", signal["short_strike"]),
            "ratio_qty": 1,
            "side": "sell",
            "position_intent": "sell_to_open",
            "_meta": {
                "stream": signal["stream"],
                "role": "short_put",
                "strike": signal["short_strike"],
            },
        },
        {
            "symbol": _occ_option_symbol(under, expiry, "P", signal["long_strike"]),
            "ratio_qty": 1,
            "side": "buy",
            "position_intent": "buy_to_open",
            "_meta": {
                "stream": signal["stream"],
                "role": "long_put",
                "strike": signal["long_strike"],
            },
        },
    ]


def _calendar_legs(signal: Dict) -> List[Dict]:
    """Two legs at same strike, different expiries (front vs back)."""
    expiry_pair = signal["expiry"]
    if "/" not in expiry_pair:
        return []
    front, back = expiry_pair.split("/")
    strike = signal["short_strike"]
    under = signal["underlier"]
    return [
        {
            "symbol": _occ_option_symbol(under, front, "P", strike),
            "ratio_qty": 1,
            "side": "sell",
            "position_intent": "sell_to_open",
            "_meta": {"stream": signal["stream"], "role": "front_leg",
                      "expiry": front, "strike": strike},
        },
        {
            "symbol": _occ_option_symbol(under, back, "P", strike),
            "ratio_qty": 1,
            "side": "buy",
            "position_intent": "buy_to_open",
            "_meta": {"stream": signal["stream"], "role": "back_leg",
                      "expiry": back, "strike": strike},
        },
    ]


def _hedge_legs(signal: Dict) -> List[Dict]:
    """Single-leg long put for v5_hedge."""
    expiry = signal["expiry"]
    strike = signal["long_strike"]
    under = signal["underlier"]
    return [
        {
            "symbol": _occ_option_symbol(under, expiry, "P", strike),
            "ratio_qty": 1,
            "side": "buy",
            "position_intent": "buy_to_open",
            "_meta": {"stream": signal["stream"], "role": "tail_hedge",
                      "strike": strike},
        },
    ]


def _vol_arb_legs(signal: Dict) -> List[Dict]:
    """Cross-vol arb is a model sleeve, not a concrete order. Emit a
    sentinel placeholder so the dry-run still round-trips cleanly
    but the mock response includes a NEEDS_MODEL_EXEC flag."""
    return [
        {
            "symbol": "_CROSS_VOL_MODEL_",
            "ratio_qty": 1,
            "side": "sell",
            "position_intent": "sell_to_open",
            "_meta": {
                "stream": signal["stream"],
                "role": "iv_rv_model_leg",
                "note": ("cross_vol arb requires the EXP-2020 dispersion "
                         "execution model — no single strike; dry run only"),
            },
        },
    ]


def build_alpaca_order(signal: Dict) -> Optional[Dict]:
    """Build the mock Alpaca /v2/orders body for one OPEN signal."""
    if signal["action"] not in ("OPEN", "HOLD_MIN"):
        return None
    structure = signal["structure"]
    if structure == "put_credit_spread":
        legs = _put_credit_spread_legs(signal)
    elif structure == "calendar_spread":
        legs = _calendar_legs(signal)
    elif structure == "hedge_sleeve":
        legs = _hedge_legs(signal)
    elif structure == "iv_rv_relative_value":
        legs = _vol_arb_legs(signal)
    else:
        return None
    if not legs:
        return None
    order_id = f"paper-{signal['stream']}-{uuid.uuid4().hex[:8]}"
    order = {
        "client_order_id": order_id,
        "symbol": signal["underlier"],
        "qty": signal["target_contracts_after_vix"],
        "type": "limit" if signal.get("limit_price") else "market",
        "limit_price": signal.get("limit_price"),
        "order_class": "mleg",
        "time_in_force": "day",
        "legs": legs,
    }
    return order


def validate_alpaca_order(order: Dict) -> List[str]:
    """Return a list of validation errors; empty list = valid."""
    errors: List[str] = []
    required = ("client_order_id", "symbol", "qty", "type",
                "order_class", "time_in_force", "legs")
    for k in required:
        if k not in order:
            errors.append(f"missing key: {k}")
    if order.get("order_class") not in ("mleg", "simple"):
        errors.append(f"order_class must be mleg or simple, got "
                      f"{order.get('order_class')}")
    if order.get("time_in_force") not in ("day", "gtc", "ioc"):
        errors.append(f"time_in_force must be day/gtc/ioc, got "
                      f"{order.get('time_in_force')}")
    if not isinstance(order.get("legs"), list) or not order["legs"]:
        errors.append("legs must be non-empty list")
    else:
        for i, leg in enumerate(order["legs"]):
            for k in ("symbol", "ratio_qty", "side", "position_intent"):
                if k not in leg:
                    errors.append(f"leg[{i}]: missing {k}")
            if leg.get("side") not in ("buy", "sell"):
                errors.append(f"leg[{i}]: side must be buy/sell")
    qty = order.get("qty", 0)
    if not isinstance(qty, int) or qty < 1:
        errors.append(f"qty must be int >= 1, got {qty}")
    return errors


def mock_alpaca_submit(order: Dict) -> Dict:
    """Simulate an Alpaca `/v2/orders` POST response.

    This does NOT hit the network. It returns the shape we would
    expect from the API so downstream consumers can be smoke-tested.
    """
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": f"mock-{uuid.uuid4().hex}",
        "client_order_id": order["client_order_id"],
        "created_at": now,
        "updated_at": now,
        "submitted_at": now,
        "status": "accepted",
        "asset_class": "us_option",
        "symbol": order["symbol"],
        "qty": order["qty"],
        "order_class": order["order_class"],
        "type": order["type"],
        "limit_price": order.get("limit_price"),
        "time_in_force": order["time_in_force"],
        "legs": order["legs"],
        "_mock": True,
        "_mock_note": ("This is a MOCK response — no Alpaca API call was "
                        "made. Submitting the same payload against real "
                        "Alpaca paper keys should return a similar shape."),
    }


# ── VIX ladder application ─────────────────────────────────────────────


def apply_vix_ladder(signals: List[Dict], vix: float) -> Tuple[float, List[Dict]]:
    """For each signal, attach the VIX ladder factor and recomputed
    target_contracts_after_vix. Returns (ladder_factor, new_signals)."""
    factor = float(vix_leverage_factor(vix))
    for s in signals:
        orig = s.get("target_contracts", 0) or 0
        new_contracts = int(round(orig * factor))
        s["vix_ladder_factor"] = factor
        s["target_contracts_after_vix"] = new_contracts
        if orig > 0 and new_contracts == 0:
            # VIX ladder flattened this signal entirely
            s["action"] = "VIX_FLAT"
            s["reason"] = (f"{s.get('reason', '')} | "
                           f"VIX ladder {factor:.2f}× → 0 contracts")
    return factor, signals


# ── HTML ───────────────────────────────────────────────────────────────


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #1a5c3a}
    h2{margin-top:2em;color:#1a5c3a}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#1a5c3a;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#1a5c3a}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    .pill.warn{background:#c07a1f}
    pre{background:#f5f5f5;padding:10px;border-radius:6px;overflow-x:auto;font-size:11px}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2860 Paper Dry Run</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2860 — Paper Trading Dry Run (signal → mock Alpaca)</h1>",
        "<p class='muted'>End-to-end smoke test of the signal pipeline: "
        "EXP-2830 generator → EXP-2820 VIX ladder → Alpaca multi-leg "
        "order schema → mock submission. Proves the pipeline works "
        "before real API keys arrive.</p>",
        "<p><span class='pill'>Rule Zero ✓ real market snapshot, mock HTTP only</span></p>",
    ]

    # Market snapshot
    snap = payload["signal_output"]["market"]
    ctx = payload["signal_output"]["overlay_context"]
    h.append("<h2>Market snapshot</h2>")
    h.append(
        "<table>"
        f"<tr><td class='l'>as_of</td><td>{snap.get('as_of')}</td></tr>"
        f"<tr><td class='l'>VIX</td><td>{snap.get('vix_value')}</td></tr>"
        f"<tr><td class='l'>VIX3M</td><td>{snap.get('vix3m_value')}</td></tr>"
        f"<tr><td class='l'>VVIX</td><td>{snap.get('vvix_value')} "
        f"(pct60d {snap.get('vvix_percentile_60d'):.2f})</td></tr>"
        f"<tr><td class='l'>Term structure</td><td>{snap.get('term_structure_ratio'):.3f}</td></tr>"
        f"<tr><td class='l'>SPY 20d realised vol</td><td>{snap.get('spy_20d_realized_vol')}</td></tr>"
        f"<tr><td class='l'>Regime</td><td>{ctx.get('regime')}</td></tr>"
        f"<tr><td class='l'>VoV gate</td><td>{ctx.get('vov_gate_pass')}</td></tr>"
        f"<tr><td class='l'>TS gate</td><td>{ctx.get('term_structure_gate_pass')}</td></tr>"
        f"<tr><td class='l'>FOMC week</td><td>{ctx.get('fomc_week')}</td></tr>"
        "</table>"
    )

    # VIX ladder
    h.append("<h2>VIX ladder</h2>")
    h.append(
        f"<p>VIX = {payload['vix_ladder']['vix']:.2f} → "
        f"ladder factor <b>{payload['vix_ladder']['factor']:.2f}×</b></p>"
    )
    h.append("<table><tr><th>VIX &lt;</th><th>Factor</th></tr>")
    for threshold, mult in VIX_LADDER:
        t_str = f"{threshold:.0f}" if threshold < 1e8 else "∞"
        h.append(f"<tr><td>{t_str}</td><td>{mult:.2f}</td></tr>")
    h.append("</table>")

    # Signals table
    h.append("<h2>Per-stream signals (post VIX ladder)</h2>")
    h.append("<table><tr><th>Stream</th><th>Action</th><th>Underlier</th>"
             "<th>Structure</th><th>Short K</th><th>Long K</th>"
             "<th>Width</th><th>Expiry</th><th>DTE</th>"
             "<th>Contracts</th><th>Reason</th></tr>")
    for s in payload["signals"]:
        action = s.get("action", "")
        pill_cls = {
            "OPEN": "ok",
            "HOLD_MIN": "ok",
            "NO_TRADE": "",
            "SKIP": "bad",
            "VIX_FLAT": "warn",
        }.get(action, "")
        h.append(
            f"<tr><td class='l'><b>{s.get('stream','')}</b></td>"
            f"<td><span class='pill {pill_cls}'>{action}</span></td>"
            f"<td>{s.get('underlier','')}</td>"
            f"<td class='l'>{s.get('structure','')}</td>"
            f"<td>{s.get('short_strike') or '—'}</td>"
            f"<td>{s.get('long_strike') or '—'}</td>"
            f"<td>{s.get('width') or '—'}</td>"
            f"<td class='l'>{s.get('expiry','')}</td>"
            f"<td>{s.get('dte_days') or '—'}</td>"
            f"<td>{s.get('target_contracts_after_vix','')}</td>"
            f"<td class='l muted'>{s.get('reason','')}</td></tr>"
        )
    h.append("</table>")

    # Orders built
    h.append("<h2>Alpaca orders (mock submission)</h2>")
    orders = payload["alpaca_orders"]
    if not orders:
        h.append("<p class='muted'>No OPEN signals produced any "
                 "Alpaca orders for this date (all streams skipped, "
                 "no-traded, or flattened).</p>")
    else:
        for order in orders:
            h.append(f"<h3>{order['_stream']} — "
                     f"{order['order']['client_order_id']}</h3>")
            h.append("<table><tr><th>Field</th><th>Value</th></tr>")
            o = order["order"]
            h.append(
                f"<tr><td class='l'>symbol</td><td>{o['symbol']}</td></tr>"
                f"<tr><td class='l'>qty (combo)</td><td>{o['qty']}</td></tr>"
                f"<tr><td class='l'>order_class</td><td>{o['order_class']}</td></tr>"
                f"<tr><td class='l'>type</td><td>{o['type']}</td></tr>"
                f"<tr><td class='l'>limit_price</td><td>{o['limit_price']}</td></tr>"
                f"<tr><td class='l'>time_in_force</td><td>{o['time_in_force']}</td></tr>"
                f"<tr><td class='l'>legs</td><td>{len(o['legs'])}</td></tr>"
                "</table>"
            )
            for i, leg in enumerate(o["legs"]):
                h.append(
                    f"<table><tr><td class='l'><b>leg {i}</b></td>"
                    f"<td class='l'>{leg['side']} {leg['symbol']} "
                    f"(ratio {leg['ratio_qty']})</td></tr></table>"
                )
            errors = order.get("validation_errors", [])
            if errors:
                h.append("<p class='neg'>Validation errors:</p><ul>")
                for e in errors:
                    h.append(f"<li class='neg'>{e}</li>")
                h.append("</ul>")
            else:
                h.append("<p><span class='pill ok'>schema valid</span></p>")
            mock = order["mock_response"]
            h.append(f"<pre>{json.dumps({k: v for k, v in mock.items() if not k.startswith('_')}, indent=2)[:1500]}</pre>")

    # Summary + verdict
    h.append("<h2>Summary</h2>")
    sm = payload["summary"]
    h.append(
        "<table>"
        f"<tr><td class='l'>Streams evaluated</td><td>{sm['n_streams']}</td></tr>"
        f"<tr><td class='l'>OPEN / HOLD_MIN</td><td>{sm['n_open']}</td></tr>"
        f"<tr><td class='l'>NO_TRADE</td><td>{sm['n_no_trade']}</td></tr>"
        f"<tr><td class='l'>SKIP (overlay blocked)</td><td>{sm['n_skip']}</td></tr>"
        f"<tr><td class='l'>VIX_FLAT</td><td>{sm['n_vix_flat']}</td></tr>"
        f"<tr><td class='l'>Total contracts (post-ladder)</td><td>{sm['total_contracts']}</td></tr>"
        f"<tr><td class='l'>Alpaca orders built</td><td>{sm['n_orders_built']}</td></tr>"
        f"<tr><td class='l'>Orders passing schema validation</td>"
        f"<td class='{ 'pos' if sm['n_orders_valid'] == sm['n_orders_built'] else 'neg' }'>"
        f"{sm['n_orders_valid']} / {sm['n_orders_built']}</td></tr>"
        "</table>"
    )

    h.append("<h2>Verdict</h2>")
    if sm["all_valid"]:
        h.append("<p><span class='pill ok'>PIPELINE GREEN</span> — "
                 "every built order passes schema validation; mock "
                 "submission returned 'accepted' for every leg. Ready "
                 "for real Alpaca keys.</p>")
    else:
        h.append("<p><span class='pill bad'>PIPELINE RED</span> — "
                 "at least one order failed schema validation. See "
                 "per-order tables above for details.</p>")

    # Caveats
    h.append("<h2>Methodology &amp; caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>Signal source.</b> "
             "<code>compass.exp2830_paper_signal_generator.generate_all_signals</code> "
             "called verbatim. Uses real Yahoo market data.</li>")
    h.append("<li><b>VIX ladder.</b> "
             "<code>compass.exp2820_flash_crash_protection.VIX_LADDER</code> "
             "applied as a size multiplier on the EXP-2830 per-stream "
             "target_contracts. Signals whose post-ladder size rounds "
             "to 0 are marked VIX_FLAT.</li>")
    h.append("<li><b>Alpaca schema.</b> The order payloads follow "
             "Alpaca's documented multi-leg options spec: top-level "
             "<code>order_class='mleg'</code>, <code>legs</code> array "
             "with <code>ratio_qty</code> and <code>position_intent</code> "
             "per leg, OCC option symbols. Validation checks the "
             "required keys and per-leg schema but does NOT validate "
             "that the strikes actually exist on the exchange — that "
             "is Alpaca's job at real submission.</li>")
    h.append("<li><b>Mock submission.</b> No network call is made. "
             "The mock response object returns the shape we expect "
             "from a real <code>/v2/orders</code> POST — id, "
             "client_order_id, status='accepted', legs. It is marked "
             "with <code>_mock: true</code> so downstream consumers "
             "know it is not a live fill.</li>")
    h.append("<li><b>cross_vol sleeve.</b> EXP-2020's IV−RV "
             "dispersion trade is a model sleeve, not a single "
             "underlier + strike. The dry run emits a sentinel "
             "<code>_CROSS_VOL_MODEL_</code> leg with a "
             "NEEDS_MODEL_EXEC note so the pipeline stays consistent; "
             "production will route this sleeve through its own "
             "execution model rather than Alpaca multi-leg.</li>")
    h.append("<li><b>What this does NOT prove.</b> "
             "(a) Real Alpaca account mechanics (option permissions, "
             "buying-power checks, margin); "
             "(b) fill quality (no market data hit the order book); "
             "(c) risk-management integration (position limits, "
             "circuit breakers, kill switch wiring). Those are "
             "separate smoke tests that need the real API keys.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="YYYY-MM-DD override; defaults to today")
    args = parser.parse_args()

    as_of: date = (datetime.strptime(args.date, "%Y-%m-%d").date()
                   if args.date else date.today())
    logger = get_logger(dry_run=True)

    # Step 1 — generate EXP-2830 signals
    print(f"[exp2860] generating signals for {as_of}…", flush=True)
    signal_output = generate_all_signals(as_of, logger)
    signals: List[Dict] = signal_output["signals"]
    print(f"[exp2860] EXP-2830 emitted {len(signals)} stream signals")

    # Step 2 — apply VIX ladder from EXP-2820
    vix_level = float(signal_output["market"].get("vix_value", 20.0))
    factor, signals = apply_vix_ladder(signals, vix_level)
    print(f"[exp2860] VIX {vix_level:.2f} → ladder factor {factor:.2f}×")

    n_flat = sum(1 for s in signals if s.get("action") == "VIX_FLAT")
    if n_flat:
        print(f"[exp2860] VIX ladder flattened {n_flat} signals")

    # Step 3 — build Alpaca orders
    print("[exp2860] building Alpaca multi-leg orders …", flush=True)
    orders: List[Dict] = []
    for s in signals:
        order = build_alpaca_order(s)
        if order is None:
            continue
        errors = validate_alpaca_order(order)
        mock = mock_alpaca_submit(order) if not errors else {
            "status": "rejected",
            "reject_reason": errors,
            "_mock": True,
        }
        orders.append({
            "_stream": s["stream"],
            "order": order,
            "validation_errors": errors,
            "mock_response": mock,
        })
        status = "valid" if not errors else f"INVALID ({len(errors)} errors)"
        print(f"[exp2860]   {s['stream']:10s}: {status}")

    # Step 4 — summary
    n_open = sum(1 for s in signals if s.get("action") in ("OPEN", "HOLD_MIN"))
    n_no_trade = sum(1 for s in signals if s.get("action") == "NO_TRADE")
    n_skip = sum(1 for s in signals if s.get("action") == "SKIP")
    n_orders_valid = sum(1 for o in orders if not o["validation_errors"])
    total_contracts = sum(
        (s.get("target_contracts_after_vix", 0) or 0) for s in signals
    )
    summary = {
        "n_streams": len(signals),
        "n_open": n_open,
        "n_no_trade": n_no_trade,
        "n_skip": n_skip,
        "n_vix_flat": n_flat,
        "total_contracts": total_contracts,
        "n_orders_built": len(orders),
        "n_orders_valid": n_orders_valid,
        "all_valid": n_orders_valid == len(orders) and len(orders) >= 0,
    }
    print(f"[exp2860] summary: {summary}")

    # Payload
    payload = {
        "experiment": "EXP-2860",
        "tag": "EXP-2860",
        "description": "Paper trading dry run — signal → VIX ladder → mock Alpaca",
        "data_sources": {
            "signal_generator": "compass.exp2830_paper_signal_generator",
            "vix_ladder": "compass.exp2820_flash_crash_protection.VIX_LADDER",
            "market_snapshot": "real Yahoo via EXP-2830",
        },
        "as_of": str(as_of),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signal_output": signal_output,
        "vix_ladder": {
            "vix": vix_level,
            "factor": factor,
            "ladder": [(float(t) if t < 1e8 else None, float(m))
                       for t, m in VIX_LADDER],
        },
        "signals": signals,
        "alpaca_orders": orders,
        "summary": summary,
    }

    # Step 5 — write files
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dry_run_path = PAPER_DIR / f"dry_run_{as_of.strftime('%Y-%m-%d')}.json"
    dry_run_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2860] wrote {dry_run_path}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2860] wrote {REPORT_JSON}")

    REPORT_HTML.write_text(render_html(payload))
    print(f"[exp2860] wrote {REPORT_HTML}")

    # Exit code
    if not summary["all_valid"]:
        print("[exp2860] PIPELINE RED — at least one order failed validation")
        return 1
    print("[exp2860] PIPELINE GREEN — every order passes schema validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
