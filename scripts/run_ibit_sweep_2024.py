#!/usr/bin/env python3
"""
run_ibit_sweep_2024.py — IBIT put credit spread parameter sweep.

Uses REAL IBIT options data from crypto_options_cache.db (Polygon OCC format).
Data range: Nov 2024 – Mar 2026. No synthetic data — Carlos directive.

Train period: Nov 2024 – Sep 2025 (~10 months)
Test period:  Oct 2025 – Mar 2026 (~6 months)

New Gate 2 criteria (Carlos, March 2026):
  - avg annualized return ≥ 50%
  - max drawdown < 25% (i.e. better than -25%)
  - overfit score ≥ 0.70  (test_annualized / train_annualized)

Strategy: Bull-put credit spreads on IBIT
  - Entry: find nearest expiration with DTE in [dte_target - 5, dte_target + 10]
  - Short put: nearest confirmed strike ≤ spot × (1 - otm_pct)
  - Long put:  nearest confirmed strike ≤ short_strike - spread_width
  - Credit filter: credit / spread_width ≥ min_credit_pct
  - Exit: profit target OR stop loss OR expiry
  - Sizing: flat-risk, n_contracts = floor(equity × risk_pct / max_loss_per_contract)

Output:
  output/ibit_sweep_2024_leaderboard.json  — all results sorted by avg_annualized_return
  output/ibit_sweep_2024_state.json        — checkpoint every 50 combos
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH       = ROOT / "data" / "crypto_options_cache.db"
OUTPUT_DIR    = ROOT / "output"
LEADERBOARD   = OUTPUT_DIR / "ibit_sweep_2024_leaderboard.json"
STATE_PATH    = OUTPUT_DIR / "ibit_sweep_2024_state.json"

# ---------------------------------------------------------------------------
# Full date range covered by real IBIT data
# ---------------------------------------------------------------------------
# Data starts Nov 1, 2024. First option expiry: Nov 22, 2024.
# We define "effective start" as Nov 1, 2024 so the equity curve starts then.

FULL_START  = "2024-11-01"
FULL_END    = "2026-03-20"
TRAIN_END   = "2025-09-30"   # ~10 months of train
TEST_START  = "2025-10-01"   # ~6 months of test

# Gate 2 criteria
GATE2_AVG_ANN_RETURN = 50.0   # annualized %
GATE2_MAX_DD         = -25.0  # max drawdown (dd is negative, so ≥ -25%)
GATE2_OVERFIT        = 0.70   # test_ann / train_ann

STARTING_CAPITAL = 100_000.0

# ---------------------------------------------------------------------------
# Fast grid (phase 1): fix spread_width=3, max_contracts=10, kelly_frac=0.5
# ---------------------------------------------------------------------------
#
# Sweep: DTE × OTM × PT × SL × IC = 4×5×4×4×2 = 640 combos
# After finding top 20, run refinement on full dimensions.
# The mission asks for dollar spread_width. We translate to a fixed dollar amount
# for the strike selection (IBIT trades $30-$72 in this window).

FAST_GRID = {
    "dte":            [7, 10, 14, 21],
    "otm_pct":        [0.03, 0.05, 0.08, 0.10, 0.15],
    "spread_width":   [3],          # fixed in fast grid (dollar amount)
    "profit_target":  [0.30, 0.50, 0.65, 0.80],
    "stop_loss_mult": [1.5, 2.0, 2.5, 3.0],
    "risk_pct":       [0.10],       # 10% per trade
    # max_contracts must be high enough to not be the binding constraint.
    # IBIT at $50, spread=$3, max_loss~$2.40/sh → $240/contract.
    # At 10% risk of $100k = $10k → need ~41 contracts. Use 100 as practical max.
    "max_contracts":  [100],
    "iron_condor":    [False, True],
}

REFINE_GRID = {
    "dte":            [7, 10, 14, 21],
    "otm_pct":        [0.03, 0.05, 0.08, 0.10, 0.15],
    "spread_width":   [3, 5, 7],
    "profit_target":  [0.30, 0.50, 0.65, 0.80],
    "stop_loss_mult": [1.5, 2.0, 2.5, 3.0],
    "risk_pct":       [0.05, 0.10, 0.15, 0.20],
    "max_contracts":  [100],
    "iron_condor":    [False, True],
}


def build_combos(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


# ---------------------------------------------------------------------------
# IBIT Backtester — uses real crypto_options_cache.db
# ---------------------------------------------------------------------------

class IBITBacktester:
    """
    Put credit spread backtester for IBIT using real Polygon options data.

    All options prices are in USD (IBIT is an equity ETF, not crypto native).
    P&L is in USD.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # State (reset each run)
        self.capital: float = 0.0
        self.starting_capital: float = 0.0
        self.open_positions: List[dict] = []
        self.trades: List[dict] = []
        self.equity_curve: List[Tuple[str, float]] = []
        self._ruin = False

    def _spot(self, dt: str) -> Optional[float]:
        """IBIT underlying close price for a given date."""
        row = self._conn.execute(
            "SELECT close FROM crypto_underlying_daily "
            "WHERE ticker='IBIT' AND date <= ? AND date != '0000-00-00' "
            "ORDER BY date DESC LIMIT 1",
            (dt,),
        ).fetchone()
        return float(row["close"]) if row and row["close"] else None

    def _trading_days(self, start: str, end: str) -> List[str]:
        """All IBIT trading days (by underlying price data) in range."""
        rows = self._conn.execute(
            "SELECT date FROM crypto_underlying_daily "
            "WHERE ticker='IBIT' AND date >= ? AND date <= ? AND date != '0000-00-00' "
            "ORDER BY date",
            (start, end),
        ).fetchall()
        return [r["date"] for r in rows]

    def _puts_for_expiry(self, expiry: str, entry_date: str) -> List[Dict]:
        """
        All put strikes with a real close price on entry_date for the given expiry.
        Returns list of dicts sorted ascending by strike.
        """
        rows = self._conn.execute(
            """
            SELECT cc.strike, cd.close AS price
            FROM crypto_option_contracts cc
            JOIN crypto_option_daily cd ON cc.contract_symbol = cd.contract_symbol
            WHERE cc.ticker = 'IBIT'
              AND cc.expiration = ?
              AND cc.option_type = 'put'
              AND cd.date = ?
              AND cd.close IS NOT NULL
              AND cd.close > 0
              AND cd.date != '0000-00-00'
            ORDER BY cc.strike ASC
            """,
            (expiry, entry_date),
        ).fetchall()
        return [{"strike": float(r["strike"]), "price": float(r["price"])} for r in rows]

    def _calls_for_expiry(self, expiry: str, entry_date: str) -> List[Dict]:
        """All call strikes with a real close price on entry_date. For iron condors."""
        rows = self._conn.execute(
            """
            SELECT cc.strike, cd.close AS price
            FROM crypto_option_contracts cc
            JOIN crypto_option_daily cd ON cc.contract_symbol = cd.contract_symbol
            WHERE cc.ticker = 'IBIT'
              AND cc.expiration = ?
              AND cc.option_type = 'call'
              AND cd.date = ?
              AND cd.close IS NOT NULL
              AND cd.close > 0
              AND cd.date != '0000-00-00'
            ORDER BY cc.strike ASC
            """,
            (expiry, entry_date),
        ).fetchall()
        return [{"strike": float(r["strike"]), "price": float(r["price"])} for r in rows]

    def _option_price(self, expiry: str, strike: float, option_type: str, dt: str) -> Optional[float]:
        """Close price for a specific option on a given date."""
        # Build OCC symbol: O:IBIT{YYMMDD}{P/C}{strike_8digits}
        dt_obj = datetime.strptime(expiry, "%Y-%m-%d")
        date_str = dt_obj.strftime("%y%m%d")
        strike_int = int(round(strike * 1000))
        opt_letter = "P" if option_type == "put" else "C"
        symbol = f"O:IBIT{date_str}{opt_letter}{strike_int:08d}"

        row = self._conn.execute(
            "SELECT close FROM crypto_option_daily "
            "WHERE contract_symbol = ? AND date = ? AND date != '0000-00-00'",
            (symbol, dt),
        ).fetchone()
        if row and row["close"] is not None and row["close"] > 0:
            return float(row["close"])
        return None

    def _available_expiries(self, entry_date: str, dte_min: int, dte_max: int) -> List[Tuple[str, int]]:
        """
        Find all expiry dates with DTE in [dte_min, dte_max] as of entry_date.
        Returns list of (expiry_str, dte) sorted by dte ascending.
        """
        today = datetime.strptime(entry_date, "%Y-%m-%d").date()
        # Look 90 days forward to cover max DTE
        end_search = (today + timedelta(days=90)).strftime("%Y-%m-%d")

        rows = self._conn.execute(
            """
            SELECT DISTINCT expiration FROM crypto_option_contracts
            WHERE ticker = 'IBIT' AND expiration > ? AND expiration <= ?
            ORDER BY expiration ASC
            """,
            (entry_date, end_search),
        ).fetchall()

        result = []
        for r in rows:
            exp_str = r["expiration"]
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte_min <= dte <= dte_max:
                result.append((exp_str, dte))
        return result

    def _find_bull_put_spread(
        self, puts: List[Dict], spot: float
    ) -> Optional[Dict]:
        """
        Find best bull-put credit spread from available puts.
        Short put: nearest strike ≤ spot × (1 - otm_pct)
        Long put:  nearest strike ≤ short_strike - spread_width
        """
        otm_pct    = self.cfg["otm_pct"]
        width      = self.cfg["spread_width"]   # dollar width
        min_credit_pct = self.cfg.get("min_credit_pct", 8.0)

        target_short = spot * (1.0 - otm_pct)
        strikes = [p["strike"] for p in puts]

        # Short leg: highest strike at or below target_short
        short_candidates = [s for s in strikes if s <= target_short * 1.02]
        if not short_candidates:
            return None
        short_strike = max(short_candidates)

        # Long leg: highest strike that is at least spread_width below short_strike.
        # We require long_strike <= short_strike - spread_width (no tolerance —
        # we want at minimum the requested dollar width).
        long_candidates = [s for s in strikes if s <= short_strike - width and s < short_strike]
        if not long_candidates:
            return None
        long_strike = max(long_candidates)

        if long_strike >= short_strike:
            return None

        # Fetch prices
        short_p = next((p["price"] for p in puts if p["strike"] == short_strike), None)
        long_p  = next((p["price"] for p in puts if p["strike"] == long_strike),  None)
        if short_p is None or long_p is None:
            return None
        if short_p <= 0 or long_p < 0 or long_p >= short_p:
            return None

        credit       = short_p - long_p   # USD per share
        spread_width = short_strike - long_strike
        credit_pct   = credit / spread_width * 100.0

        if credit_pct < min_credit_pct:
            return None

        return {
            "type":         "bull_put",
            "short_strike": short_strike,
            "long_strike":  long_strike,
            "short_price":  short_p,
            "long_price":   long_p,
            "credit":       credit,
            "spread_width": spread_width,
            "credit_pct":   credit_pct,
            "max_loss":     spread_width - credit,
        }

    def _find_bear_call_spread(
        self, calls: List[Dict], spot: float
    ) -> Optional[Dict]:
        """
        Find best bear-call credit spread from available calls.
        Short call: nearest strike ≥ spot × (1 + otm_pct)
        Long call:  nearest strike ≥ short_strike + spread_width
        """
        otm_pct    = self.cfg["otm_pct"]
        width      = self.cfg["spread_width"]
        min_credit_pct = self.cfg.get("min_credit_pct", 8.0)

        target_short = spot * (1.0 + otm_pct)
        strikes = [c["strike"] for c in calls]

        # Short leg: lowest strike at or above target_short
        short_candidates = [s for s in strikes if s >= target_short * 0.98]
        if not short_candidates:
            return None
        short_strike = min(short_candidates)

        # Long leg: lowest strike that is at least spread_width above short_strike.
        long_candidates = [s for s in strikes if s >= short_strike + width and s > short_strike]
        if not long_candidates:
            return None
        long_strike = min(long_candidates)

        if long_strike <= short_strike:
            return None

        short_p = next((c["price"] for c in calls if c["strike"] == short_strike), None)
        long_p  = next((c["price"] for c in calls if c["strike"] == long_strike),  None)
        if short_p is None or long_p is None:
            return None
        if short_p <= 0 or long_p < 0 or long_p >= short_p:
            return None

        credit       = short_p - long_p
        spread_width = long_strike - short_strike
        credit_pct   = credit / spread_width * 100.0

        if credit_pct < min_credit_pct:
            return None

        return {
            "type":         "bear_call",
            "short_strike": short_strike,
            "long_strike":  long_strike,
            "short_price":  short_p,
            "long_price":   long_p,
            "credit":       credit,
            "spread_width": spread_width,
            "credit_pct":   credit_pct,
            "max_loss":     spread_width - credit,
        }

    def _try_enter(self, entry_date: str, expiry: str) -> Optional[dict]:
        """
        Attempt to open a credit spread position.
        Returns position dict or None.
        """
        spot = self._spot(entry_date)
        if not spot:
            return None

        today     = datetime.strptime(entry_date, "%Y-%m-%d").date()
        exp_date  = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte       = (exp_date - today).days

        iron_condor = self.cfg.get("iron_condor", False)

        puts  = self._puts_for_expiry(expiry, entry_date)
        calls = self._calls_for_expiry(expiry, entry_date) if iron_condor else []

        # Always enter bull-put (primary leg)
        bp = self._find_bull_put_spread(puts, spot)
        if bp is None:
            return None

        # Iron condor: add bear-call on top
        bc = None
        if iron_condor and calls:
            bc = self._find_bear_call_spread(calls, spot)
            # IC is optional — if no valid bear-call, proceed with bull-put only

        # Combined credit and max loss
        total_credit    = bp["credit"]
        total_max_loss  = bp["max_loss"]
        if bc:
            total_credit   += bc["credit"]
            total_max_loss += bc["max_loss"]

        if total_max_loss <= 0:
            return None

        # Position sizing: max_loss is per-share; multiply by MULTIPLIER to get per-contract USD
        MULTIPLIER = 100  # IBIT options = 100 shares/contract
        max_loss_per_contract_usd = total_max_loss * MULTIPLIER

        account_base = self.capital if self.cfg.get("compound", True) else self.starting_capital
        risk_budget  = account_base * self.cfg["risk_pct"]
        n_contracts  = int(risk_budget / max_loss_per_contract_usd)
        n_contracts  = max(1, min(n_contracts, self.cfg["max_contracts"]))

        position = {
            "entry_date":      entry_date,
            "expiry":          expiry,
            "dte_at_entry":    dte,
            "spot_at_entry":   spot,
            "n_contracts":     n_contracts,
            "multiplier":      MULTIPLIER,
            "iron_condor":     bc is not None,

            # Bull-put legs
            "bp_short_strike": bp["short_strike"],
            "bp_long_strike":  bp["long_strike"],
            "bp_short_price":  bp["short_price"],
            "bp_long_price":   bp["long_price"],
            "bp_credit":       bp["credit"],
            "bp_spread_width": bp["spread_width"],
            "bp_max_loss":     bp["max_loss"],
            "bp_credit_pct":   bp["credit_pct"],

            # Bear-call legs (None if no IC)
            "bc_short_strike": bc["short_strike"] if bc else None,
            "bc_long_strike":  bc["long_strike"]  if bc else None,
            "bc_short_price":  bc["short_price"]  if bc else None,
            "bc_long_price":   bc["long_price"]   if bc else None,
            "bc_credit":       bc["credit"]        if bc else 0.0,
            "bc_spread_width": bc["spread_width"]  if bc else 0.0,
            "bc_max_loss":     bc["max_loss"]      if bc else 0.0,

            # Combined totals (per share)
            "total_credit":    total_credit,
            "total_max_loss":  total_max_loss,
            "credit_pct":      total_credit / (bp["spread_width"] + (bc["spread_width"] if bc else 0)) * 100.0,

            # Exit levels
            "profit_target_price": total_credit * (1.0 - self.cfg["profit_target"]),
            "stop_loss_price":     total_credit * self.cfg["stop_loss_mult"],

            "status":          "open",
            "unrealized_pnl":  0.0,
        }

        return position

    def _current_spread_value(self, pos: dict, dt: str) -> Optional[float]:
        """
        Current value of the combined spread (cost to close = what we pay).
        For a credit spread we originally received credit; the "value" is how
        much the spread is worth now (0 = max profit, max_loss = full loss).
        """
        expiry = pos["expiry"]

        # Bull-put current value
        bp_short = self._option_price(expiry, pos["bp_short_strike"], "put", dt)
        bp_long  = self._option_price(expiry, pos["bp_long_strike"],  "put", dt)
        if bp_short is None or bp_long is None:
            return None
        bp_val = max(0.0, bp_short - bp_long)

        # Bear-call current value (if IC)
        bc_val = 0.0
        if pos["iron_condor"] and pos["bc_short_strike"] is not None:
            bc_short = self._option_price(expiry, pos["bc_short_strike"], "call", dt)
            bc_long  = self._option_price(expiry, pos["bc_long_strike"],  "call", dt)
            if bc_short is None or bc_long is None:
                # If call prices unavailable, skip today (keep holding)
                return None
            bc_val = max(0.0, bc_short - bc_long)

        return bp_val + bc_val

    def _check_exit(self, pos: dict, current_date: str) -> Optional[Tuple[float, str]]:
        """Returns (pnl_usd, reason) or None to hold."""
        spread_val = self._current_spread_value(pos, current_date)
        if spread_val is None:
            return None  # No data today — hold

        # Unrealized P&L update
        pnl_per_share = pos["total_credit"] - spread_val
        pos["unrealized_pnl"] = pnl_per_share * pos["n_contracts"] * pos["multiplier"]

        reason = None
        if spread_val <= pos["profit_target_price"]:
            reason = "profit_target"
        elif spread_val >= pos["stop_loss_price"]:
            reason = "stop_loss"

        if reason:
            pnl_usd = pnl_per_share * pos["n_contracts"] * pos["multiplier"]
            return pnl_usd, reason
        return None

    def _close_at_expiry(self, pos: dict, expiry_str: str) -> Tuple[float, str]:
        """Close at expiry using intrinsic value, falling back to real prices."""
        spot = self._spot(expiry_str) or pos["spot_at_entry"]

        # Try real prices first
        spread_val = self._current_spread_value(pos, expiry_str)
        if spread_val is not None:
            pnl_per_share = pos["total_credit"] - spread_val
            return pnl_per_share * pos["n_contracts"] * pos["multiplier"], "expiry_real"

        # Fallback: intrinsic value
        bp_intrinsic = max(0.0, pos["bp_short_strike"] - spot) - max(0.0, pos["bp_long_strike"] - spot)
        bc_intrinsic = 0.0
        if pos["iron_condor"] and pos["bc_short_strike"] is not None:
            bc_intrinsic = (max(0.0, spot - pos["bc_short_strike"])
                          - max(0.0, spot - pos["bc_long_strike"]))
        spread_val = max(0.0, bp_intrinsic) + max(0.0, bc_intrinsic)
        pnl_per_share = pos["total_credit"] - spread_val
        return pnl_per_share * pos["n_contracts"] * pos["multiplier"], "expiry_intrinsic"

    def _record_close(self, pos: dict, exit_date: str, pnl_usd: float, reason: str):
        self.capital += pnl_usd
        self.trades.append({
            "entry_date":      pos["entry_date"],
            "exit_date":       exit_date,
            "expiry":          pos["expiry"],
            "dte_at_entry":    pos["dte_at_entry"],
            "bp_short_strike": pos["bp_short_strike"],
            "bp_long_strike":  pos["bp_long_strike"],
            "bp_credit":       pos["bp_credit"],
            "bc_credit":       pos["bc_credit"],
            "total_credit":    pos["total_credit"],
            "n_contracts":     pos["n_contracts"],
            "spot_at_entry":   pos["spot_at_entry"],
            "spot_at_exit":    self._spot(exit_date) or pos["spot_at_entry"],
            "pnl_usd":         pnl_usd,
            "exit_reason":     reason,
            "win":             pnl_usd > 0,
            "iron_condor":     pos["iron_condor"],
        })
        pos["status"] = "closed"

    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run backtest from start_date to end_date.
        Allows max 1 open position at a time.
        """
        self.capital          = STARTING_CAPITAL
        self.starting_capital = STARTING_CAPITAL
        self.open_positions   = []
        self.trades           = []
        self.equity_curve     = []
        self._ruin            = False

        dte_target = self.cfg["dte"]
        dte_min    = max(2, dte_target - 5)
        dte_max    = dte_target + 10

        trading_days = self._trading_days(start_date, end_date)

        for current_date in trading_days:
            if self._ruin:
                break

            today = datetime.strptime(current_date, "%Y-%m-%d").date()

            # 1. Close expired positions
            for pos in list(self.open_positions):
                if pos["status"] != "open":
                    continue
                exp_date = datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
                if today >= exp_date:
                    pnl_usd, reason = self._close_at_expiry(pos, pos["expiry"])
                    self._record_close(pos, pos["expiry"], pnl_usd, reason)

            self.open_positions = [p for p in self.open_positions if p["status"] == "open"]

            # 2. Check stop-loss / profit-target on open positions
            for pos in list(self.open_positions):
                result = self._check_exit(pos, current_date)
                if result:
                    pnl_usd, reason = result
                    self._record_close(pos, current_date, pnl_usd, reason)

            self.open_positions = [p for p in self.open_positions if p["status"] == "open"]

            # 3. Entry: one position at a time
            if len(self.open_positions) == 0 and not self._ruin:
                expiries = self._available_expiries(current_date, dte_min, dte_max)
                for exp_str, dte in expiries:
                    pos = self._try_enter(current_date, exp_str)
                    if pos:
                        self.open_positions.append(pos)
                        break  # one entry per day

            # 4. Equity curve
            unrealized = sum(p.get("unrealized_pnl", 0.0) for p in self.open_positions)
            total_eq   = self.capital + unrealized
            self.equity_curve.append((current_date, total_eq))

            if self.capital <= 0:
                self._ruin = True

        return self._build_results(start_date, end_date)

    def _build_results(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Build results dict with annualized return and max drawdown."""
        import math

        trades    = self.trades
        n_trades  = len(trades)

        # --- Return ---
        final_cap = self.capital
        return_pct = (final_cap - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0

        # Annualize: use calendar days between start and end
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
        days     = max(1, (end_dt - start_dt).days)
        years    = days / 365.25
        ann_return = ((final_cap / STARTING_CAPITAL) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

        # --- Max drawdown ---
        if self.equity_curve:
            equities = [e for _, e in self.equity_curve]
            peak     = equities[0]
            max_dd   = 0.0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (eq - peak) / peak * 100.0
                if dd < max_dd:
                    max_dd = dd
        else:
            max_dd = 0.0

        # --- Trade stats ---
        wins    = [t for t in trades if t["win"]]
        losers  = [t for t in trades if not t["win"]]
        win_rate   = len(wins) / n_trades * 100.0 if n_trades > 0 else 0.0
        avg_win    = sum(t["pnl_usd"] for t in wins)    / len(wins)   if wins   else 0.0
        avg_loss   = sum(abs(t["pnl_usd"]) for t in losers) / len(losers) if losers else 0.0
        gross_win  = sum(t["pnl_usd"] for t in wins)    if wins   else 0.0
        gross_loss = sum(t["pnl_usd"] for t in losers)  if losers else 0.0
        profit_factor = abs(gross_win / gross_loss) if gross_loss != 0 else 999.0

        return {
            "start_date":     start_date,
            "end_date":       end_date,
            "days":           days,
            "years":          round(years, 3),
            "return_pct":     round(return_pct, 2),
            "ann_return":     round(ann_return, 2),
            "max_drawdown":   round(max_dd, 2),
            "total_trades":   n_trades,
            "win_rate":       round(win_rate, 2),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "profit_factor":  round(profit_factor, 4),
            "starting_capital": STARTING_CAPITAL,
            "ending_capital": round(final_cap, 2),
            "ruin":           self._ruin,
            "trades":         trades,
            "equity_curve":   [{"date": d, "equity": e} for d, e in self.equity_curve],
        }

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Sweep harness
# ---------------------------------------------------------------------------

def run_backtest(params: Dict[str, Any], start: str, end: str) -> Optional[Dict]:
    """Run backtest for a parameter set over a date range. Returns None on error."""
    cfg = {
        "dte":           params["dte"],
        "otm_pct":       params["otm_pct"],
        "spread_width":  params["spread_width"],
        "profit_target": params["profit_target"],
        "stop_loss_mult":params["stop_loss_mult"],
        "risk_pct":      params["risk_pct"],
        "max_contracts": params["max_contracts"],
        "iron_condor":   params["iron_condor"],
        "compound":      True,
        "min_credit_pct": 5.0,   # at least 5% of spread width
    }
    try:
        bt = IBITBacktester(cfg)
        result = bt.run(start, end)
        bt.close()
        return result
    except Exception as e:
        return None


def annualize(return_pct: float, days: int) -> float:
    """Convert a total return % over N days to annualized %."""
    if days <= 0:
        return 0.0
    years = days / 365.25
    multiplier = (1.0 + return_pct / 100.0)
    if multiplier <= 0:
        return -100.0
    ann = (multiplier ** (1.0 / years) - 1.0) * 100.0
    return round(ann, 2)


def compute_overfit(
    train_ann: float,
    test_ann: float,
) -> float:
    """overfit_score = test_ann / train_ann. Clamped to [-1, 2]."""
    if abs(train_ann) < 0.01:
        return 0.0
    score = test_ann / train_ann
    return round(max(-1.0, min(2.0, score)), 4)


def gate2_check(avg_ann: float, max_dd: float, overfit: float) -> bool:
    return (
        avg_ann  >= GATE2_AVG_ANN_RETURN
        and max_dd >= GATE2_MAX_DD
        and overfit >= GATE2_OVERFIT
    )


def build_record(
    combo_idx: int,
    params: Dict,
    full_res: Dict,
    train_res: Dict,
    test_res: Dict,
    overfit: float,
) -> Dict:
    """Build a leaderboard record."""
    avg_ann = full_res["ann_return"]
    max_dd  = full_res["max_drawdown"]

    train_ann = train_res["ann_return"] if train_res else 0.0
    test_ann  = test_res["ann_return"]  if test_res  else 0.0

    pass2 = gate2_check(avg_ann, max_dd, overfit)

    return {
        "combo_idx":   combo_idx,
        "params":      params,
        "ann_return":  avg_ann,
        "max_drawdown": max_dd,
        "win_rate":    full_res["win_rate"],
        "profit_factor": full_res["profit_factor"],
        "total_trades": full_res["total_trades"],
        "return_pct":  full_res["return_pct"],
        "train_ann":   round(train_ann, 2),
        "test_ann":    round(test_ann, 2),
        "overfit_score": overfit,
        "gate2_pass":  pass2,
        "start_date":  full_res["start_date"],
        "end_date":    full_res["end_date"],
        "years":       full_res["years"],
    }


def load_leaderboard() -> List[Dict]:
    if LEADERBOARD.exists():
        try:
            with open(LEADERBOARD) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def save_leaderboard(records: List[Dict]):
    sorted_records = sorted(records, key=lambda r: r.get("ann_return", -999), reverse=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD, "w") as f:
        json.dump(sorted_records, f, indent=2)


def save_state(combo_idx: int, n_done: int, n_gate2: int, elapsed: float, phase: str, total: int):
    state = {
        "phase":              phase,
        "last_combo_idx":     combo_idx,
        "n_completed":        n_done,
        "total_combos":       total,
        "n_gate2_pass":       n_gate2,
        "elapsed_seconds":    round(elapsed, 1),
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def run_sweep(
    combos: List[Dict],
    phase: str,
    checkpoint: int = 50,
    existing_records: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Run a sweep over param combos. Returns updated records list.
    """
    records = list(existing_records or [])
    existing_keys = {json.dumps(r["params"], sort_keys=True) for r in records}

    n_done  = 0
    n_gate2 = sum(1 for r in records if r.get("gate2_pass"))
    t_start = time.time()

    total = len(combos)

    # Full period: Nov 2024 – Mar 2026
    # Train: Nov 2024 – Sep 2025
    # Test: Oct 2025 – Mar 2026
    FULL_S  = FULL_START
    FULL_E  = FULL_END
    TRAIN_S = FULL_START
    TRAIN_E = TRAIN_END
    TEST_S  = TEST_START
    TEST_E  = FULL_END

    print(f"\n{'='*70}")
    print(f"IBIT Sweep — Phase: {phase}  |  {total} combos")
    print(f"Full period:  {FULL_S} → {FULL_E}")
    print(f"Train period: {TRAIN_S} → {TRAIN_E}")
    print(f"Test period:  {TEST_S} → {TEST_E}")
    print(f"Gate 2: ann_return ≥ {GATE2_AVG_ANN_RETURN}%  |  "
          f"max_dd ≥ {GATE2_MAX_DD}%  |  overfit ≥ {GATE2_OVERFIT}")
    print(f"{'='*70}\n")

    for combo_idx, params in enumerate(combos):
        param_key = json.dumps(params, sort_keys=True)

        # Skip if already done
        if param_key in existing_keys:
            n_done += 1
            continue

        # --- Full period ---
        full_res = run_backtest(params, FULL_S, FULL_E)
        if full_res is None or full_res.get("total_trades", 0) == 0:
            n_done += 1
            continue

        # --- Train / test ---
        train_res = run_backtest(params, TRAIN_S, TRAIN_E)
        test_res  = run_backtest(params, TEST_S, TEST_E)

        train_ann = train_res["ann_return"] if train_res else 0.0
        test_ann  = test_res["ann_return"]  if test_res  else 0.0

        overfit = compute_overfit(train_ann, test_ann)
        record  = build_record(combo_idx, params, full_res, train_res, test_res, overfit)
        records.append(record)
        existing_keys.add(param_key)
        n_done += 1

        gate2_marker = " *** GATE2 PASS ***" if record["gate2_pass"] else ""
        if record["gate2_pass"]:
            n_gate2 += 1

        elapsed  = time.time() - t_start
        rate     = n_done / elapsed if elapsed > 0 else 0
        remaining = (total - n_done) / rate if rate > 0 else 0

        print(
            f"[{combo_idx:4d}/{total}] "
            f"DTE={params['dte']:2d} "
            f"OTM={params['otm_pct']*100:.0f}% "
            f"W=${params['spread_width']:2.0f} "
            f"PT={params['profit_target']*100:.0f}% "
            f"SL={params['stop_loss_mult']:.1f}x "
            f"R={params['risk_pct']*100:.0f}% "
            f"IC={'Y' if params['iron_condor'] else 'N'}  "
            f"ann={record['ann_return']:+7.1f}% "
            f"dd={record['max_drawdown']:+6.1f}% "
            f"ovf={overfit:.2f} "
            f"tr={full_res['total_trades']:3d} "
            f"wr={record['win_rate']:.0f}%"
            f"{gate2_marker}"
        )

        # Checkpoint
        if n_done % checkpoint == 0 or n_done == total:
            save_leaderboard(records)
            save_state(combo_idx, n_done, n_gate2, elapsed, phase, total)
            top3 = sorted(records, key=lambda r: r.get("ann_return", -999), reverse=True)[:3]
            print(f"\n  --- Checkpoint {combo_idx}/{total} | "
                  f"Gate2: {n_gate2} | ETA: {remaining/60:.1f}min ---")
            for rank, r in enumerate(top3, 1):
                p = r["params"]
                print(f"  #{rank}: DTE={p['dte']} OTM={p['otm_pct']*100:.0f}% "
                      f"W=${p['spread_width']} PT={p['profit_target']*100:.0f}% "
                      f"SL={p['stop_loss_mult']:.1f}x  "
                      f"ann={r['ann_return']:+.1f}% dd={r['max_drawdown']:+.1f}% "
                      f"ovf={r['overfit_score']:.2f} gate2={r['gate2_pass']}")
            print()

    # Final save
    elapsed = time.time() - t_start
    save_leaderboard(records)
    save_state(total - 1, n_done, n_gate2, elapsed, phase, total)

    print(f"\n{'='*70}")
    print(f"Phase {phase} COMPLETE | {n_done}/{total} done | {n_gate2} Gate2 passes")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"{'='*70}")

    return records


def print_final_summary(records: List[Dict]):
    """Print Gate 2 passes and top 10 closest misses."""
    gate2  = sorted([r for r in records if r.get("gate2_pass")],
                    key=lambda r: r["ann_return"], reverse=True)
    all_r  = sorted(records, key=lambda r: r.get("ann_return", -999), reverse=True)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS — {len(records)} combos evaluated")
    print(f"Gate 2 passes: {len(gate2)}")
    print(f"{'='*70}")

    if gate2:
        print(f"\nGATE 2 CHAMPIONS (ann_return ≥ {GATE2_AVG_ANN_RETURN}%, "
              f"max_dd ≥ {GATE2_MAX_DD}%, overfit ≥ {GATE2_OVERFIT}):\n")
        hdr = (f"{'Rank':<5} {'DTE':>4} {'OTM%':>5} {'W$':>4} {'PT%':>4} "
               f"{'SL':>4} {'Rsk%':>5} {'IC':>3}  "
               f"{'AnnRet':>8} {'MaxDD':>7} {'Ovf':>5} {'WR%':>5} {'Tr':>4}")
        print(hdr)
        print("-" * len(hdr))
        for rank, r in enumerate(gate2[:20], 1):
            p = r["params"]
            print(
                f"{rank:<5} {p['dte']:>4} {p['otm_pct']*100:>4.0f}% "
                f"{p['spread_width']:>4.0f} {p['profit_target']*100:>3.0f}% "
                f"{p['stop_loss_mult']:>4.1f} {p['risk_pct']*100:>4.0f}%  "
                f"{'Y' if p['iron_condor'] else 'N':>3}  "
                f"{r['ann_return']:>+8.1f}% {r['max_drawdown']:>+7.1f}% "
                f"{r['overfit_score']:>5.2f} {r['win_rate']:>5.1f}% {r['total_trades']:>4}"
            )
        print()

        # Per-trade details for top 3
        if gate2:
            print("Train / Test breakdown for top champion:")
            r = gate2[0]
            print(f"  Full ann:   {r['ann_return']:+.1f}%")
            print(f"  Train ann:  {r['train_ann']:+.1f}%")
            print(f"  Test ann:   {r['test_ann']:+.1f}%")
            print(f"  Overfit:    {r['overfit_score']:.2f}")
            print(f"  Max DD:     {r['max_drawdown']:+.1f}%")
            print(f"  Win rate:   {r['win_rate']:.1f}%")
            print(f"  Profit factor: {r['profit_factor']:.2f}")

    else:
        print("\nNo Gate 2 passes found.")
        print("\nTop 10 closest misses:")
        hdr = (f"{'Rank':<5} {'DTE':>4} {'OTM%':>5} {'W$':>4} {'PT%':>4} "
               f"{'SL':>4} {'Rsk%':>5} {'IC':>3}  "
               f"{'AnnRet':>8} {'MaxDD':>7} {'Ovf':>5} {'WR%':>5}")
        print(hdr)
        print("-" * len(hdr))
        for rank, r in enumerate(all_r[:10], 1):
            p = r["params"]
            # Flag what failed
            flags = []
            if r["ann_return"] < GATE2_AVG_ANN_RETURN:
                flags.append(f"ret={r['ann_return']:+.1f}%<{GATE2_AVG_ANN_RETURN}%")
            if r["max_drawdown"] < GATE2_MAX_DD:
                flags.append(f"dd={r['max_drawdown']:+.1f}%")
            if r["overfit_score"] < GATE2_OVERFIT:
                flags.append(f"ovf={r['overfit_score']:.2f}")
            print(
                f"{rank:<5} {p['dte']:>4} {p['otm_pct']*100:>4.0f}% "
                f"{p['spread_width']:>4.0f} {p['profit_target']*100:>3.0f}% "
                f"{p['stop_loss_mult']:>4.1f} {p['risk_pct']*100:>4.0f}%  "
                f"{'Y' if p['iron_condor'] else 'N':>3}  "
                f"{r['ann_return']:>+8.1f}% {r['max_drawdown']:>+7.1f}% "
                f"{r['overfit_score']:>5.2f} {r['win_rate']:>5.1f}%"
                f"  FAIL: {', '.join(flags)}"
            )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="IBIT credit spread parameter sweep (real data)")
    parser.add_argument("--phase", choices=["fast", "refine", "both"], default="both",
                        help="Sweep phase: fast grid, refinement, or both (default: both)")
    parser.add_argument("--checkpoint", type=int, default=50,
                        help="Checkpoint every N combos (default: 50)")
    parser.add_argument("--top-n-refine", type=int, default=20,
                        help="Top N fast-grid winners to refine around (default: 20)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_leaderboard()

    if args.phase in ("fast", "both"):
        fast_combos = build_combos(FAST_GRID)
        print(f"\nPhase 1 — Fast grid: {len(fast_combos)} combos")
        records = run_sweep(fast_combos, phase="fast",
                            checkpoint=args.checkpoint,
                            existing_records=records)

    if args.phase in ("refine", "both"):
        # Build refinement combos
        refine_combos = build_combos(REFINE_GRID)
        print(f"\nPhase 2 — Refinement grid: {len(refine_combos)} combos")
        records = run_sweep(refine_combos, phase="refine",
                            checkpoint=args.checkpoint,
                            existing_records=records)

    print_final_summary(records)
    print(f"\nLeaderboard: {LEADERBOARD}")
    print(f"State:       {STATE_PATH}")


if __name__ == "__main__":
    main()
