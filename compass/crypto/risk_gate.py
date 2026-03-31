"""
Risk gate for the crypto ETF scanner.

All entry risk checks are collected here and exposed through a single
``check_all_gates(context)`` call.  Every gate returns a standardised dict:

    {
        "gate":    str,    # gate name
        "blocked": bool,
        "reason":  str,    # human-readable explanation (empty when not blocked)
        ...                # gate-specific diagnostic fields
    }

``check_all_gates`` aggregates them:

    {
        "allowed": bool,           # True only if ALL gates pass
        "gates":   list[dict],     # one entry per gate that was evaluated
    }

Context dict keys (all optional — missing keys skip the corresponding gate):

    btc_price_now        float   current BTC/ETF price
    etf_close_price      float   prior-session close price for gap check
    current_iv           float   current IV (annualised decimal)
    historical_ivs       list    historical IV series for percentile rank
    day_of_week          int     Python weekday: 0=Mon … 6=Sun
    dte                  int     days-to-expiration of the candidate trade

Gate thresholds (class constants — override via ``RiskGate.THRESHOLD_*``):

    GAP_BLOCK_PCT         5.0%    max tolerated overnight gap
    IV_PERCENTILE_FLOOR  40.0     minimum IV percentile to enter
    WEEKEND_DTE_MAX       7       DTE above which Thu/Fri is OK to enter
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Thresholds (module-level so tests can patch them trivially)
# ---------------------------------------------------------------------------

GAP_BLOCK_PCT: float = 5.0          # block if |gap| > this %
IV_PERCENTILE_FLOOR: float = 40.0   # block if IV percentile < this
WEEKEND_DTE_MAX: int = 7            # block Thu/Fri entries with DTE ≤ this


# ---------------------------------------------------------------------------
# Individual gate functions
# ---------------------------------------------------------------------------

def check_overnight_gap(
    btc_price_now: float,
    etf_close_price: float,
) -> Dict[str, Any]:
    """Block if the overnight gap between current price and prior close > 5%.

    A large gap means the ETF opened far from its modelled value — slippage
    and mispricing risk are elevated.

    Args:
        btc_price_now:    Current BTC or crypto ETF price.
        etf_close_price:  Prior-session close of the same instrument.

    Returns:
        {
            "gate":      "overnight_gap",
            "blocked":   bool,
            "gap_pct":   float,  # signed %, positive = gapped up
            "reason":    str,
        }

    Raises:
        ValueError: If etf_close_price is zero.
    """
    if etf_close_price == 0:
        raise ValueError("etf_close_price must be non-zero.")

    gap_pct = (btc_price_now - etf_close_price) / etf_close_price * 100.0
    abs_gap = abs(gap_pct)
    blocked = abs_gap > GAP_BLOCK_PCT

    return {
        "gate":    "overnight_gap",
        "blocked": blocked,
        "gap_pct": round(gap_pct, 4),
        "reason":  (
            f"Gap {gap_pct:+.2f}% exceeds ±{GAP_BLOCK_PCT}% threshold"
            if blocked else ""
        ),
    }


def check_iv_percentile(
    current_iv: float,
    historical_ivs: List[float],
) -> Dict[str, Any]:
    """Block if the current IV is below the 40th percentile of history.

    Entering when IV is cheap (low percentile) means we collect thin premiums
    relative to realised vol risk — the edge disappears.

    Args:
        current_iv:      Current implied volatility (any consistent unit).
        historical_ivs:  Historical IV series (same unit, any length ≥ 1).

    Returns:
        {
            "gate":       "iv_percentile",
            "blocked":    bool,
            "percentile": float,   # 0-100
            "reason":     str,
        }

    Raises:
        ValueError: If historical_ivs is empty.
    """
    if not historical_ivs:
        raise ValueError("historical_ivs must contain at least one value.")

    n_below = sum(1 for h in historical_ivs if h <= current_iv)
    percentile = n_below / len(historical_ivs) * 100.0
    blocked = percentile < IV_PERCENTILE_FLOOR

    return {
        "gate":       "iv_percentile",
        "blocked":    blocked,
        "percentile": round(percentile, 2),
        "reason":     (
            f"IV percentile {percentile:.1f} is below floor {IV_PERCENTILE_FLOOR}"
            if blocked else ""
        ),
    }


def check_weekend_risk(day_of_week: int, dte: int) -> Dict[str, Any]:
    """Block new short-dated entries on Thu/Fri.

    A 7-DTE trade entered Thursday expires the following Thursday — it carries
    the full weekend gap risk of both the coming Saturday/Sunday and the next
    one.  Friday entries are even worse.  Block both for DTE ≤ 7.

    Args:
        day_of_week: Python ``datetime.weekday()`` value: 0=Mon … 6=Sun.
        dte:         Days-to-expiration of the candidate trade.

    Returns:
        {
            "gate":        "weekend_risk",
            "blocked":     bool,
            "day_of_week": int,
            "dte":         int,
            "reason":      str,
        }
    """
    is_thu_or_fri = day_of_week in (3, 4)  # Thursday=3, Friday=4
    is_short_dte = dte <= WEEKEND_DTE_MAX
    blocked = is_thu_or_fri and is_short_dte

    day_name = {3: "Thursday", 4: "Friday"}.get(day_of_week, f"weekday {day_of_week}")

    return {
        "gate":        "weekend_risk",
        "blocked":     blocked,
        "day_of_week": day_of_week,
        "dte":         dte,
        "reason":      (
            f"{day_name} entry with DTE={dte} ≤ {WEEKEND_DTE_MAX} carries "
            f"unhedgeable weekend gap risk"
            if blocked else ""
        ),
    }


# ---------------------------------------------------------------------------
# Aggregate gate runner
# ---------------------------------------------------------------------------

def check_all_gates(context: Dict[str, Any]) -> Dict[str, Any]:
    """Run all applicable risk gates from a context dictionary.

    Gates are skipped (not counted as blocked) when their required keys are
    absent from ``context`` — this allows partial contexts in unit tests and
    in production scans that lack certain data feeds.

    Args:
        context: Dict that may contain any of:
            btc_price_now   (float)  — required for overnight_gap gate
            etf_close_price (float)  — required for overnight_gap gate
            current_iv      (float)  — required for iv_percentile gate
            historical_ivs  (list)   — required for iv_percentile gate
            day_of_week     (int)    — required for weekend_risk gate
            dte             (int)    — required for weekend_risk gate

    Returns:
        {
            "allowed": bool,         # True only when every evaluated gate passes
            "gates":   list[dict],   # one result dict per gate evaluated
        }
    """
    gates: List[Dict[str, Any]] = []

    # --- Overnight gap ---
    if "btc_price_now" in context and "etf_close_price" in context:
        gates.append(
            check_overnight_gap(
                btc_price_now=float(context["btc_price_now"]),
                etf_close_price=float(context["etf_close_price"]),
            )
        )

    # --- IV percentile ---
    if "current_iv" in context and "historical_ivs" in context:
        gates.append(
            check_iv_percentile(
                current_iv=float(context["current_iv"]),
                historical_ivs=list(context["historical_ivs"]),
            )
        )

    # --- Weekend risk ---
    if "day_of_week" in context and "dte" in context:
        gates.append(
            check_weekend_risk(
                day_of_week=int(context["day_of_week"]),
                dte=int(context["dte"]),
            )
        )

    allowed = all(not g["blocked"] for g in gates)

    return {
        "allowed": allowed,
        "gates":   gates,
    }
