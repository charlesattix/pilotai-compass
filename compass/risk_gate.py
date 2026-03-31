"""
Risk Gate — hard-coded risk rules from the MASTERPLAN.

Every alert must pass through ``RiskGate.check()`` before dispatch.
The rules are intentionally NOT configurable; they use constants from
``shared/constants.py`` which are hard-coded per MASTERPLAN §Risk Management.

COMPASS extensions (config-driven, default OFF):
  - Rule 8: macro score sizing flags (fear/greed) — never blocks, flags only.
  - Rule 9: RRG quadrant filter — blocks bull-put alerts for sector ETFs in
    Lagging/Weakening quadrant when ``compass.rrg_filter: true``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alerts.alert_schema import Alert

from shared.constants import (
    COOLDOWN_AFTER_STOP,
    DAILY_LOSS_LIMIT,
    MAX_CORRELATED_POSITIONS,
    MAX_RISK_PER_TRADE,
    MAX_TOTAL_EXPOSURE,
    WEEKLY_LOSS_LIMIT,
)

logger = logging.getLogger(__name__)


def _directions_match(pos_direction: str, alert_direction_value: str) -> bool:
    """Compare a position's stored direction string against an alert's Direction enum value.

    Positions are stored by strategy_type ("bull_put_spread", "bear_call_spread", …).
    Alert directions use the Direction enum ("bullish", "bearish", "neutral").
    Both representations are normalised to a common set before comparison.
    """
    _BULL = {"bullish", "bull", "bull_put", "bull_put_spread"}
    _BEAR = {"bearish", "bear", "bear_call", "bear_call_spread"}
    _NEUTRAL = {"neutral", "iron_condor"}

    pos_norm = pos_direction.lower()
    alert_norm = alert_direction_value.lower()

    if alert_norm in _BULL:
        return pos_norm in _BULL
    if alert_norm in _BEAR:
        return pos_norm in _BEAR
    if alert_norm in _NEUTRAL:
        return pos_norm in _NEUTRAL
    return False


class RiskGate:
    """Hard-coded risk gate with optional config-driven drawdown circuit breaker."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        risk_cfg = self.config.get("risk", {})
        # BUG #19 fix: MAX_TOTAL_EXPOSURE is configurable; MASTERPLAN default 15%
        # Fallback chain: risk.max_total_exposure_pct
        #   -> risk.portfolio_risk.max_portfolio_risk_pct
        #   -> risk.max_portfolio_risk_pct
        #   -> MAX_TOTAL_EXPOSURE constant (15%)
        cfg_pct = (
            risk_cfg.get("max_total_exposure_pct")
            or risk_cfg.get("portfolio_risk", {}).get("max_portfolio_risk_pct")
            or risk_cfg.get("max_portfolio_risk_pct")
        )
        self._max_total_exposure = (float(cfg_pct) / 100.0) if cfg_pct is not None else MAX_TOTAL_EXPOSURE
        # Per-trade risk cap: read from config.risk.max_risk_per_trade (default: constant 5%)
        cfg_per_trade = risk_cfg.get("max_risk_per_trade")
        self._max_risk_per_trade = (float(cfg_per_trade) / 100.0) if cfg_per_trade is not None else MAX_RISK_PER_TRADE

    def check(self, alert: Alert, account_state: dict) -> tuple:
        """Evaluate an alert against all risk rules.

        Args:
            alert: The candidate alert.
            account_state: Dict with shape::

                {
                    "account_value": float,
                    "open_positions": list[dict],
                        # each has: ticker, direction, risk_pct, entry_time
                    "daily_pnl_pct": float,
                    "weekly_pnl_pct": float,
                    "recent_stops": list[dict],
                        # each has: ticker, stopped_at (datetime)
                    "circuit_breaker": bool,  # True when broker API is down
                }

        Returns:
            ``(True, "")`` if approved.
            ``(False, reason)`` if rejected.
        """
        # 0. Circuit breaker — Alpaca API down; block ALL new trades
        if account_state.get("circuit_breaker", False):
            reason = "circuit_breaker=True — Alpaca account state unavailable, halting new entries"
            logger.warning("RiskGate BLOCKED: %s", reason)
            return (False, reason)

        # 1. Per-trade risk cap
        if alert.risk_pct > self._max_risk_per_trade:
            reason = (
                f"Per-trade risk {alert.risk_pct:.2%} exceeds "
                f"max {self._max_risk_per_trade:.2%}"
            )
            logger.warning("RiskGate BLOCKED: %s", reason)
            return (False, reason)

        # 2. Total exposure (epsilon for float precision)
        open_risk = sum(
            p.get("risk_pct", 0) for p in account_state.get("open_positions", [])
        )
        if open_risk + alert.risk_pct > self._max_total_exposure + 1e-9:
            reason = (
                f"Total exposure would be {open_risk + alert.risk_pct:.2%}, "
                f"exceeds max {self._max_total_exposure:.2%}"
            )
            logger.warning("RiskGate BLOCKED: %s", reason)
            return (False, reason)

        # 3. Daily loss limit — stop all alerts for the day
        daily_pnl_pct = account_state.get("daily_pnl_pct", 0.0)
        if daily_pnl_pct < -DAILY_LOSS_LIMIT:
            reason = (
                f"Daily P&L {daily_pnl_pct:.2%} below limit "
                f"-{DAILY_LOSS_LIMIT:.2%} — no more alerts today"
            )
            logger.warning("RiskGate BLOCKED: %s", reason)
            return (False, reason)

        # 4. Weekly loss limit — flag for 50% size reduction (does not block)
        #    We set a flag on the alert but do NOT reject here.
        #    The position sizer will read this flag later.
        #    (checked externally; this method just logs)
        weekly_pnl_pct = account_state.get("weekly_pnl_pct", 0.0)
        if weekly_pnl_pct < -WEEKLY_LOSS_LIMIT:
            logger.info(
                "RiskGate FLAG: weekly P&L %s below -%s — 50%% size reduction",
                f"{weekly_pnl_pct:.2%}",
                f"{WEEKLY_LOSS_LIMIT:.2%}",
            )

        # 5. Correlated positions (same direction)
        # Uses _directions_match() to normalise between DB storage values
        # ("bull_put_spread") and Alert Direction enum values ("bullish").
        same_direction_count = sum(
            1
            for p in account_state.get("open_positions", [])
            if _directions_match(p.get("direction", ""), alert.direction.value)
        )
        if same_direction_count >= MAX_CORRELATED_POSITIONS:
            reason = (
                f"Already {same_direction_count} open {alert.direction.value} "
                f"positions (max {MAX_CORRELATED_POSITIONS})"
            )
            logger.warning("RiskGate BLOCKED: %s", reason)
            return (False, reason)

        # 5.5 Per-ticker position limit — mirrors backtester max_positions_per_ticker.
        #     Prevents the live scanner from opening unlimited positions in one underlying
        #     across multiple expirations (e.g. 3 SPY iron condors in one scan).
        #     Config: risk.max_positions_per_ticker (int).  Not enforced if absent.
        max_per_ticker = self.config.get("risk", {}).get("max_positions_per_ticker")
        if max_per_ticker is not None:
            ticker_positions = sum(
                1 for p in account_state.get("open_positions", [])
                if p.get("ticker", "").upper() == alert.ticker.upper()
            )
            if ticker_positions >= int(max_per_ticker):
                reason = (
                    f"{alert.ticker}: already {ticker_positions} open position(s) "
                    f"— max {max_per_ticker} per ticker"
                )
                logger.warning("RiskGate BLOCKED: %s", reason)
                return (False, reason)

        # 6. Cooldown after stop-out on same ticker
        now = datetime.now(timezone.utc)
        for stop in account_state.get("recent_stops", []):
            if stop.get("ticker") != alert.ticker:
                continue
            stopped_at = stop.get("stopped_at")
            if isinstance(stopped_at, str):
                try:
                    stopped_at = datetime.fromisoformat(stopped_at)
                except (ValueError, TypeError):
                    logger.warning("RiskGate: unparseable stopped_at '%s', skipping cooldown check", stopped_at)
                    continue
            if stopped_at and (now - stopped_at).total_seconds() < COOLDOWN_AFTER_STOP:
                remaining = COOLDOWN_AFTER_STOP - (now - stopped_at).total_seconds()
                reason = (
                    f"{alert.ticker} stopped out recently — "
                    f"{remaining:.0f}s remaining in cooldown"
                )
                logger.warning("RiskGate BLOCKED: %s", reason)
                return (False, reason)

        # 7. Drawdown circuit breaker (config-driven, default 35%)
        drawdown_cb_pct = self.config.get("risk", {}).get("drawdown_cb_pct", 0)
        if drawdown_cb_pct > 0:
            account_value = account_state.get("account_value", 0)
            starting_capital = self.config.get("backtest", {}).get(
                "starting_capital",
                self.config.get("risk", {}).get("account_size", account_value),
            )
            # Mirror backtester behavior (backtester.py lines 679-683):
            #   flat/non-compound mode → compare against starting_capital (no ratchet).
            #   compound mode          → compare against peak_equity (high-water mark).
            sizing_mode = self.config.get("risk", {}).get("sizing_mode", "dynamic")
            compound = self.config.get("backtest", {}).get("compound", True)
            use_flat = (sizing_mode == "flat") or (not compound)
            if use_flat:
                cb_reference = starting_capital
            else:
                cb_reference = account_state.get("peak_equity", starting_capital)
            if cb_reference and cb_reference > 0:
                drawdown_pct = ((account_value - cb_reference) / cb_reference) * 100
                if drawdown_pct < -drawdown_cb_pct:
                    reason = (
                        f"Drawdown CB triggered: {drawdown_pct:.1f}% < "
                        f"-{drawdown_cb_pct}% — halting new entries"
                    )
                    logger.warning("RiskGate BLOCKED: %s", reason)
                    return (False, reason)

        # 7.5. VIX entry gate — hard block when VIX exceeds vix_max_entry.
        # Mirrors backtester: strategy_params.get('vix_max_entry', 0); 0 = disabled.
        # current_vix must be present in account_state (populated by _build_account_state).
        vix_max_entry = self.config.get("strategy", {}).get("vix_max_entry", 0)
        if vix_max_entry > 0:
            current_vix = account_state.get("current_vix", 0)
            if current_vix > vix_max_entry:
                reason = (
                    f"VIX {current_vix:.1f} > vix_max_entry {vix_max_entry} "
                    "— entry blocked"
                )
                logger.warning("RiskGate BLOCKED: %s", reason)
                return (False, reason)

        # ── COMPASS extensions (config-driven, default OFF) ──────────────────

        # 8. Macro score sizing flags.
        #    Score < 45 (fear)  → boost sizing (flag only, never blocks).
        #    Score > 75 (greed) → reduce sizing (flag only, never blocks).
        #    The 'macro_sizing_flag' key in account_state communicates this to
        #    the downstream position sizer (CC2).
        macro_sizing_flag = account_state.get('macro_sizing_flag')
        if macro_sizing_flag == 'boost':
            macro_score = account_state.get('macro_score', 50.0)
            logger.info(
                "RiskGate COMPASS: macro_score=%.1f (fear) — boost sizing flag active",
                macro_score,
            )
        elif macro_sizing_flag == 'reduce':
            macro_score = account_state.get('macro_score', 50.0)
            logger.info(
                "RiskGate COMPASS: macro_score=%.1f (greed) — reduce sizing flag active",
                macro_score,
            )

        # 9. RRG quadrant filter (only when compass.rrg_filter: true).
        #    Blocks bull-put alerts for sector ETFs currently in Lagging or
        #    Weakening quadrant — prevents chasing decelerating relative strength.
        compass_cfg = self.config.get('compass', {})
        if compass_cfg.get('rrg_filter', False) and alert.direction.value == 'bullish':
            rrg_quadrants: dict = account_state.get('rrg_quadrants', {})
            quadrant = rrg_quadrants.get(alert.ticker, '')
            if quadrant in ('Lagging', 'Weakening'):
                reason = (
                    f"{alert.ticker} RRG quadrant is '{quadrant}' — "
                    "bull-put blocked by COMPASS RRG filter"
                )
                logger.warning("RiskGate BLOCKED: %s", reason)
                return (False, reason)

        # 10. COMPASS portfolio risk limits (only when compass.portfolio_mode: true).
        #     Enforces per-ticker sector cap, correlated-group cap, and max
        #     total directional exposure across the multi-underlying portfolio.
        if compass_cfg.get('portfolio_mode', False):
            passed, reason = self._check_compass_portfolio_limits(
                alert, account_state, compass_cfg
            )
            if not passed:
                logger.warning("RiskGate BLOCKED (COMPASS portfolio limit): %s", reason)
                return (False, reason)

        return (True, "")

    def _check_compass_portfolio_limits(
        self,
        alert: Alert,
        account_state: dict,
        compass_cfg: dict,
    ) -> tuple:
        """Enforce COMPASS portfolio risk limits.

        Checks (all configurable, all default to permissive when absent):
          a) Max single sector exposure (no sector > max_single_sector_pct)
          b) Correlated sector group cap (e.g., tech cluster combined)
          c) Max total directional delta exposure across portfolio

        Returns:
            (True, "") if approved.
            (False, reason) if rejected.
        """
        limits_cfg = compass_cfg.get("portfolio_risk_limits", {})
        account_value = account_state.get("account_value", 0)
        open_positions = account_state.get("open_positions", [])
        incoming_ticker = alert.ticker.upper()

        if account_value <= 0:
            return (True, "")

        # a) Max single sector pct
        max_sector_pct = float(limits_cfg.get("max_single_sector_pct", 0.40))
        ticker_risk_pct = sum(
            p.get("risk_pct", 0)
            for p in open_positions
            if p.get("ticker", "").upper() == incoming_ticker
        )
        if ticker_risk_pct >= max_sector_pct:
            return (
                False,
                f"{incoming_ticker} sector exposure {ticker_risk_pct:.1%} "
                f">= max {max_sector_pct:.1%}",
            )

        # b) Correlated sector group cap
        for group_name, group_cfg in limits_cfg.get("correlated_sector_groups", {}).items():
            group_tickers = [t.upper() for t in group_cfg.get("tickers", [])]
            if incoming_ticker not in group_tickers:
                continue
            max_combined_pct = float(group_cfg.get("max_combined_pct", 1.0))
            group_risk_pct = sum(
                p.get("risk_pct", 0)
                for p in open_positions
                if p.get("ticker", "").upper() in group_tickers
            )
            if group_risk_pct >= max_combined_pct:
                return (
                    False,
                    f"Correlated group '{group_name}' exposure {group_risk_pct:.1%} "
                    f">= max {max_combined_pct:.1%} (adding {incoming_ticker})",
                )

        # c) Max total portfolio delta (directional positions only)
        max_total_delta_pct = float(limits_cfg.get("max_total_delta_pct", 1.0))
        if max_total_delta_pct < 1.0:
            directional_risk = sum(
                p.get("risk_pct", 0)
                for p in open_positions
                if p.get("direction", "") not in ("neutral", "")
            )
            incoming_is_directional = alert.direction.value not in ("neutral",)
            incoming_risk_contribution = alert.risk_pct if incoming_is_directional else 0.0
            total_directional = directional_risk + incoming_risk_contribution
            if total_directional > max_total_delta_pct:
                return (
                    False,
                    f"Total directional exposure {total_directional:.1%} "
                    f"> max {max_total_delta_pct:.1%}",
                )

        return (True, "")

    def weekly_loss_breach(self, account_state: dict) -> bool:
        """Return True if the weekly loss limit has been breached."""
        return account_state.get("weekly_pnl_pct", 0.0) < -WEEKLY_LOSS_LIMIT
