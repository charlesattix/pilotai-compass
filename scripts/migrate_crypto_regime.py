"""
migrate_crypto_regime.py — Add crypto_regime table to macro_state.db

Idempotent: safe to run multiple times. Uses the versioned migration system
in shared/macro_state_db.py (schema_version 3).

Usage:
    python3 scripts/migrate_crypto_regime.py
    python3 scripts/migrate_crypto_regime.py --deploy   # also migrate deploy copy
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

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

from shared.macro_state_db import get_db, get_state, init_db


# Canonical DDL — single source of truth for both the migration and this script.
CRYPTO_REGIME_DDL = """
    CREATE TABLE IF NOT EXISTS crypto_regime (
        snapshot_date       TEXT PRIMARY KEY,
        btc_price           REAL,
        eth_price           REAL,
        fear_greed_value    INTEGER,
        fear_greed_class    TEXT,
        btc_funding_rate    REAL,
        eth_funding_rate    REAL,
        btc_realized_vol_7d  REAL,
        btc_realized_vol_30d REAL,
        btc_iv_percentile   REAL,
        btc_dominance       REAL,
        btc_put_call_ratio  REAL,
        composite_score     REAL,
        score_band          TEXT,
        ma200_position      TEXT,
        overnight_gap_pct   REAL,
        created_at          TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_crypto_regime_date ON crypto_regime(snapshot_date);
"""


def _migrate(db_path: Optional[str] = None) -> None:
    """Ensure crypto_regime table exists in the target DB."""
    # init_db runs versioned migrations (including v3 which creates crypto_regime)
    init_db(db_path)

    # Verify the table is present
    conn = get_db(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM sqlite_master "
            "WHERE type='table' AND name='crypto_regime'"
        ).fetchone()
        if row["n"] == 0:
            logger.warning(
                "crypto_regime table not found after init_db — applying DDL directly"
            )
            conn.executescript(CRYPTO_REGIME_DDL)
            conn.commit()
            logger.info("crypto_regime table created directly.")
        else:
            logger.info("crypto_regime table confirmed present.")

        # Report row count
        count = conn.execute("SELECT COUNT(*) AS n FROM crypto_regime").fetchone()["n"]
        logger.info("  Existing rows: %d", count)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate macro_state.db: add crypto_regime table")
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Also migrate the deploy copy at deploy/macro-api/data/macro_state.db",
    )
    args = parser.parse_args()

    # Migrate main DB
    main_db = str(PROJECT_ROOT / "data" / "macro_state.db")
    logger.info("Migrating main DB: %s", main_db)
    _migrate(main_db)

    if args.deploy:
        deploy_db = str(PROJECT_ROOT / "deploy" / "macro-api" / "data" / "macro_state.db")
        logger.info("Migrating deploy DB: %s", deploy_db)
        _migrate(deploy_db)

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
