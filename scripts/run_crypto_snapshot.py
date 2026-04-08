"""
Crypto Snapshot CLI
===================
Fetches all crypto market signals, computes the composite regime score,
and stores the result in the crypto_regime table in macro_state.db.

Modes:
  --daily     Fetch all signals and save one snapshot for today.
              Run once per day after US market open (~10 AM ET).

  --backtest  (Reserved for future use) — backfill historical snapshots.

Usage:
  python3 scripts/run_crypto_snapshot.py --daily
  python3 scripts/run_crypto_snapshot.py --daily --dry-run
"""

import argparse
import logging
import sys
from datetime import date, timezone, datetime
from pathlib import Path

from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Compass collectors ────────────────────────────────────────────────────────
from compass.crypto import coingecko, deribit, fear_greed, funding_rates
from compass.crypto import realized_vol as rv_module
from compass.crypto import regime as regime_module
from compass.crypto.composite_score import compute_composite_score

# ── DB helpers ────────────────────────────────────────────────────────────────
from shared.macro_state_db import init_db, save_crypto_regime


# ─────────────────────────────────────────────────────────────────────────────
# Signal collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_signals() -> dict:
    """Fetch all available crypto signals. Returns a raw signal dict.

    Each field is None if the upstream API call failed — callers must handle
    optional values. No exceptions are raised.
    """
    signals: dict = {}

    # 1. Fear & Greed
    logger.info("Fetching Fear & Greed index...")
    fg = fear_greed.get_current()
    signals["fear_greed_value"] = fg["value"] if fg else None
    signals["fear_greed_class"] = fg["classification"] if fg else None
    if fg:
        logger.info("  Fear & Greed: %d (%s)", fg["value"], fg["classification"])
    else:
        logger.warning("  Fear & Greed: unavailable")

    # 2. BTC/ETH spot prices
    logger.info("Fetching BTC/ETH prices...")
    signals["btc_price"] = coingecko.get_btc_price()
    signals["eth_price"] = coingecko.get_eth_price()
    logger.info(
        "  BTC: %s  ETH: %s",
        f"${signals['btc_price']:,.0f}" if signals["btc_price"] else "unavailable",
        f"${signals['eth_price']:,.0f}" if signals["eth_price"] else "unavailable",
    )

    # 3. BTC dominance
    logger.info("Fetching BTC dominance...")
    signals["btc_dominance"] = coingecko.get_btc_dominance()
    if signals["btc_dominance"] is not None:
        logger.info("  BTC dominance: %.1f%%", signals["btc_dominance"])
    else:
        logger.warning("  BTC dominance: unavailable")

    # 4. BTC price history — used for realized vol and MA200
    logger.info("Fetching BTC price history (365 days)...")
    history = coingecko.get_btc_history(days=365)
    btc_closes = [bar["close"] for bar in history if bar.get("close")]

    # 5. Realized volatility
    signals["btc_realized_vol_7d"] = None
    signals["btc_realized_vol_30d"] = None
    if len(btc_closes) >= 31:
        try:
            signals["btc_realized_vol_7d"] = round(
                rv_module.compute_realized_vol(btc_closes, window=7), 4
            )
            signals["btc_realized_vol_30d"] = round(
                rv_module.compute_realized_vol(btc_closes, window=30), 4
            )
            logger.info(
                "  Realized vol: 7d=%.1f%%  30d=%.1f%%",
                (signals["btc_realized_vol_7d"] or 0) * 100,
                (signals["btc_realized_vol_30d"] or 0) * 100,
            )
        except Exception as exc:
            logger.warning("  Realized vol: failed (%s)", exc)
    else:
        logger.warning("  Not enough BTC history for realized vol (%d bars)", len(btc_closes))

    # 6. MA200 position
    signals["ma200_position"] = regime_module.compute_ma200_position(btc_closes)
    logger.info("  MA200 position: %s", signals["ma200_position"])

    # 7. Overnight gap (BTC daily close-to-close as proxy for IBIT/ETHA gap)
    signals["overnight_gap_pct"] = None
    if len(btc_closes) >= 2:
        signals["overnight_gap_pct"] = round(
            (btc_closes[-1] / btc_closes[-2]) - 1.0, 5
        )
        logger.info(
            "  Overnight gap (BTC): %+.2f%%",
            (signals["overnight_gap_pct"] or 0) * 100,
        )

    # 8. Funding rates
    logger.info("Fetching funding rates...")
    signals["btc_funding_rate"] = funding_rates.get_btc_funding()
    signals["eth_funding_rate"] = funding_rates.get_eth_funding()
    logger.info(
        "  Funding: BTC=%s  ETH=%s",
        f"{signals['btc_funding_rate']:.4f}%" if signals["btc_funding_rate"] is not None else "unavailable",
        f"{signals['eth_funding_rate']:.4f}%" if signals["eth_funding_rate"] is not None else "unavailable",
    )

    # 9. BTC put/call ratio (Deribit public API — may be slow or unavailable)
    logger.info("Fetching BTC put/call ratio from Deribit...")
    signals["btc_put_call_ratio"] = deribit.get_btc_put_call_ratio()
    if signals["btc_put_call_ratio"] is not None:
        logger.info("  BTC P/C ratio: %.3f", signals["btc_put_call_ratio"])
    else:
        logger.warning("  BTC P/C ratio: unavailable")

    # 10. BTC IV percentile — requires Deribit DVOL historical data (paid/complex)
    #     TODO: implement via deribit.get_volatility_index_data() when DVOL history
    #     is available. For now, this field stays None and the composite score
    #     engine degrades gracefully by re-weighting remaining signals.
    signals["btc_iv_percentile"] = None

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(signals: dict) -> dict:
    """Compute the composite regime score from raw signals.

    Returns the signals dict augmented with composite_score and score_band.
    """
    try:
        result = compute_composite_score(
            fear_greed_index=signals.get("fear_greed_value"),
            funding_rate=signals.get("btc_funding_rate"),
            ma200_position=signals.get("ma200_position"),
            btc_dominance=signals.get("btc_dominance"),
            put_call_ratio=signals.get("btc_put_call_ratio"),
        )
        signals["composite_score"] = result["score"]
        signals["score_band"] = result["band"]
        logger.info(
            "  Composite score: %.1f  band: %s",
            result["score"],
            result["band"],
        )
    except ValueError as exc:
        # All inputs were None
        logger.warning("composite_score: all signals missing — %s", exc)
        signals["composite_score"] = None
        signals["score_band"] = None

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(snap: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  CRYPTO REGIME SNAPSHOT — {snap['snapshot_date']}")
    print("=" * 60)

    if snap.get("btc_price"):
        print(f"  BTC:  ${snap['btc_price']:>12,.0f}")
    if snap.get("eth_price"):
        print(f"  ETH:  ${snap['eth_price']:>12,.2f}")
    print()

    if snap.get("composite_score") is not None:
        print(f"  COMPOSITE SCORE: {snap['composite_score']:.1f}/100  [{snap.get('score_band', 'N/A')}]")
    if snap.get("ma200_position"):
        print(f"  MA200 position:  {snap['ma200_position'].upper()}")
    if snap.get("fear_greed_value") is not None:
        print(f"  Fear & Greed:    {snap['fear_greed_value']}  ({snap.get('fear_greed_class', '')})")
    if snap.get("btc_dominance") is not None:
        print(f"  BTC dominance:   {snap['btc_dominance']:.1f}%")
    if snap.get("btc_funding_rate") is not None:
        print(f"  BTC funding:     {snap['btc_funding_rate']:.4f}%/8h")
    if snap.get("btc_realized_vol_7d") is not None:
        print(
            f"  Realized vol:    7d={snap['btc_realized_vol_7d']*100:.1f}%"
            f"  30d={snap.get('btc_realized_vol_30d', 0)*100:.1f}%"
        )
    if snap.get("btc_put_call_ratio") is not None:
        print(f"  BTC P/C ratio:   {snap['btc_put_call_ratio']:.3f}")
    if snap.get("overnight_gap_pct") is not None:
        print(f"  Overnight gap:   {snap['overnight_gap_pct']*100:+.2f}%")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Daily runner
# ─────────────────────────────────────────────────────────────────────────────

def run_daily(dry_run: bool = False) -> None:
    """Fetch all signals, compute score, and save snapshot for today."""
    today = date.today().isoformat()
    logger.info("Running crypto snapshot for %s (dry_run=%s)", today, dry_run)

    if not dry_run:
        init_db()

    # Collect and score
    signals = collect_signals()
    signals = compute_score(signals)
    signals["snapshot_date"] = today

    _print_report(signals)

    if dry_run:
        logger.info("[DRY RUN] Snapshot not saved.")
        return

    save_crypto_regime(signals)
    logger.info("Snapshot saved to crypto_regime table (date=%s)", today)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crypto Regime Snapshot CLI")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--daily", action="store_true", help="Fetch and store today's snapshot")
    p.add_argument("--dry-run", action="store_true", help="Print snapshot without saving to DB")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.daily:
        run_daily(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
