"""
Macro State Database
====================
Production SQLite store for macro intelligence state.

Tables (defined in macro_system_architecture.md):
  snapshots     — weekly snapshot header
  sector_rs     — per-sector RS per snapshot
  macro_score   — 4-dimension macro score history
  macro_events  — FOMC / CPI / NFP calendar (updated daily)
  macro_state   — key-value current state

Integration API (read-only, called by scanner/backtester):
  get_current_macro_score()   -> float
  get_sector_rankings()       -> list[dict]
  get_event_scaling_factor()  -> float
  get_eligible_underlyings()  -> list[str]
"""

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from shared.constants import DATA_DIR

logger = logging.getLogger(__name__)

MACRO_DB_PATH = Path(DATA_DIR) / "macro_state.db"

# Schema version — increment when adding columns/indexes via MIGRATIONS
CURRENT_SCHEMA_VERSION = 2

# Keys are TARGET versions. Migration runs when stored schema_version < key.
MIGRATIONS: Dict[int, str] = {
    2: """
        ALTER TABLE macro_score ADD COLUMN overall_v2 REAL;
        ALTER TABLE macro_score ADD COLUMN score_velocity REAL;
        ALTER TABLE macro_score ADD COLUMN risk_app_velocity REAL;
        ALTER TABLE macro_score ADD COLUMN updated_at TEXT;
        ALTER TABLE macro_events ADD COLUMN is_emergency INTEGER DEFAULT 0;
        CREATE INDEX IF NOT EXISTS idx_macro_score_date ON macro_score(date);
    """,
}

# ── SQLite pragmas ────────────────────────────────────────────────────────────
SQLITE_BUSY_TIMEOUT_MS = 5000     # wait up to 5 s for locked database
SQLITE_WAL_AUTOCHECKPOINT = 1000  # checkpoint every N pages

# ── Macro regime thresholds ──────────────────────────────────────────────────
MACRO_BULL_THRESHOLD = 65   # overall score >= this → BULL_MACRO
MACRO_BEAR_THRESHOLD = 45   # overall score < this  → BEAR_MACRO

# ── Integration API defaults ────────────────────────────────────────────────
NEUTRAL_SCORE_FALLBACK = 50.0  # returned when no macro data available
DEFAULT_STALENESS_DAYS = 10    # warn if most recent snapshot is older

# ── Universe expansion ───────────────────────────────────────────────────────
SECTOR_RANK_CUTOFF = 4  # top/bottom N sectors eligible for universe expansion

# Sectors eligible for universe expansion (liquid options, >300K daily options vol)
LIQUID_SECTOR_ETFS = ["XLE", "XLF", "XLV", "XLK", "XLI", "XLU", "XLY"]
BASE_UNIVERSE = ["SPY", "QQQ", "IWM"]


# ─────────────────────────────────────────────────────────────────────────────
# Connection & schema
# ─────────────────────────────────────────────────────────────────────────────

def get_db(path: Optional[str] = None) -> sqlite3.Connection:
    """Return a WAL-mode connection to macro_state.db.

    D8: WAL mode is set on every connection. Callers should close connections
    promptly; long-running processes should call wal_checkpoint() periodically
    to prevent the WAL file from growing unbounded.
    """
    db_path = Path(path) if path else MACRO_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA wal_autocheckpoint={SQLITE_WAL_AUTOCHECKPOINT}")
    conn.row_factory = sqlite3.Row
    return conn


def wal_checkpoint(path: Optional[str] = None) -> None:
    """D8: Trigger a WAL checkpoint to flush the WAL file into the main DB.

    Call from long-running processes (e.g., after each weekly snapshot batch)
    to prevent the WAL file from accumulating unbounded write history.
    """
    conn = get_db(path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
        logger.debug("WAL checkpoint completed for macro_state.db")
    finally:
        conn.close()


def init_db(path: Optional[str] = None) -> None:
    """Create all tables if they don't exist, then run pending migrations."""
    conn = get_db(path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                date            TEXT PRIMARY KEY,
                spy_close       REAL,
                top_sector_3m   TEXT,
                top_sector_12m  TEXT,
                leading_sectors TEXT,
                lagging_sectors TEXT,
                macro_overall   REAL,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sector_rs (
                date         TEXT NOT NULL,
                ticker       TEXT NOT NULL,
                name         TEXT,
                category     TEXT,
                close        REAL,
                rs_3m        REAL,
                rs_12m       REAL,
                rs_ratio     REAL,
                rs_momentum  REAL,
                rrg_quadrant TEXT,
                rank_3m      INTEGER,
                rank_12m     INTEGER,
                PRIMARY KEY (date, ticker)
            );

            CREATE TABLE IF NOT EXISTS macro_score (
                date              TEXT PRIMARY KEY,
                overall           REAL,
                growth            REAL,
                inflation         REAL,
                fed_policy        REAL,
                risk_appetite     REAL,
                regime            TEXT,
                cfnai_3m          REAL,
                payrolls_3m_avg_k REAL,
                cpi_yoy_pct       REAL,
                core_cpi_yoy_pct  REAL,
                breakeven_5y      REAL,
                t10y2y            REAL,
                fedfunds          REAL,
                vix               REAL,
                hy_oas_pct        REAL
            );

            CREATE TABLE IF NOT EXISTS macro_events (
                event_date     TEXT NOT NULL,
                event_type     TEXT NOT NULL,
                description    TEXT,
                days_out       INTEGER,
                scaling_factor REAL,
                PRIMARY KEY (event_date, event_type)
            );

            CREATE TABLE IF NOT EXISTS macro_state (
                key        TEXT PRIMARY KEY,
                value      TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_sector_rs_date ON sector_rs(date);
            CREATE INDEX IF NOT EXISTS idx_macro_events_date ON macro_events(event_date);
        """)
        conn.commit()
        logger.debug("macro_state.db base schema verified")
    finally:
        conn.close()
    # Run any pending migrations to bring schema up to current version
    migrate_db(path)


def migrate_db(path: Optional[str] = None) -> None:
    """Run all pending schema migrations in version order.

    Migration keys are TARGET versions. A migration runs when the stored
    schema_version < target_version. Migrations are idempotent — ALTER TABLE
    failures (column already exists) are caught and logged.
    The applied version is tracked in macro_state.schema_version.
    """
    current = int(get_state("schema_version", default="0", db_path=path) or 0)
    if current >= CURRENT_SCHEMA_VERSION:
        return

    conn = get_db(path)
    try:
        for target_version in sorted(MIGRATIONS.keys()):
            if current >= target_version:
                continue  # already at or past this version
            for stmt in MIGRATIONS[target_version].strip().split(";"):
                stmt = stmt.strip()
                if not stmt or stmt.startswith("--"):
                    continue
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    # Column/index may already exist on partial migrations — skip
                    logger.debug("Migration v%d stmt skipped (%s): %s", target_version, e, stmt[:60])
            conn.commit()
            conn.execute(
                "INSERT OR REPLACE INTO macro_state (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("schema_version", str(target_version)),
            )
            conn.commit()
            logger.info("macro_state.db migrated to v%d", target_version)
            current = target_version
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────────────────────────────────────

def _macro_regime(overall: Optional[float]) -> str:
    if overall is None:
        return "NEUTRAL_MACRO"
    if overall >= MACRO_BULL_THRESHOLD:
        return "BULL_MACRO"
    if overall < MACRO_BEAR_THRESHOLD:
        return "BEAR_MACRO"
    return "NEUTRAL_MACRO"


def save_snapshot(snap: dict, db_path: Optional[str] = None) -> None:
    """
    Persist a full snapshot dict (as returned by MacroSnapshotEngine.generate_snapshot)
    into macro_state.db.  Uses INSERT OR REPLACE so re-running is idempotent.
    """
    conn = get_db(db_path)
    try:
        snap_date = snap["date"]
        ms = snap.get("macro_score") or {}
        ind = ms.get("indicators") or {}

        # snapshots table
        conn.execute(
            """
            INSERT OR REPLACE INTO snapshots
              (date, spy_close, top_sector_3m, top_sector_12m,
               leading_sectors, lagging_sectors, macro_overall)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snap_date,
                snap.get("spy_close"),
                snap.get("top_sector_3m"),
                snap.get("top_sector_12m"),
                json.dumps(snap.get("leading_sectors") or []),
                json.dumps(snap.get("lagging_sectors") or []),
                ms.get("overall"),
            ),
        )

        # sector_rs table
        for item in snap.get("sector_rankings") or []:
            conn.execute(
                """
                INSERT OR REPLACE INTO sector_rs
                  (date, ticker, name, category, close, rs_3m, rs_12m,
                   rs_ratio, rs_momentum, rrg_quadrant, rank_3m, rank_12m)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap_date,
                    item["ticker"],
                    item.get("name"),
                    item.get("category"),
                    item.get("close"),
                    item.get("rs_3m"),
                    item.get("rs_12m"),
                    item.get("rs_ratio"),
                    item.get("rs_momentum"),
                    item.get("rrg_quadrant"),
                    item.get("rank_3m"),
                    item.get("rank_12m"),
                ),
            )

        # Compute score_velocity and risk_app_velocity from week-over-week delta
        # if the engine hasn't already populated them in the snapshot dict.
        score_velocity = ms.get("score_velocity")
        risk_app_velocity = ms.get("risk_app_velocity")
        if score_velocity is None or risk_app_velocity is None:
            prev = conn.execute(
                "SELECT overall, risk_appetite FROM macro_score WHERE date < ? ORDER BY date DESC LIMIT 1",
                (snap_date,),
            ).fetchone()
            if prev:
                if score_velocity is None and ms.get("overall") is not None and prev["overall"] is not None:
                    score_velocity = round(float(ms["overall"]) - float(prev["overall"]), 2)
                if risk_app_velocity is None and ms.get("risk_appetite") is not None and prev["risk_appetite"] is not None:
                    risk_app_velocity = round(float(ms["risk_appetite"]) - float(prev["risk_appetite"]), 2)
            else:
                # First-ever row: no prior week, use 0.0 (cleaner than NULL for consumers)
                if score_velocity is None:
                    score_velocity = 0.0
                if risk_app_velocity is None:
                    risk_app_velocity = 0.0

        # overall_v2: copy of overall (reserved for future alternative formula)
        overall_v2 = ms.get("overall")

        # macro_score table — includes velocity columns (score_velocity, risk_app_velocity)
        conn.execute(
            """
            INSERT OR REPLACE INTO macro_score
              (date, overall, overall_v2, growth, inflation, fed_policy, risk_appetite, regime,
               cfnai_3m, payrolls_3m_avg_k, cpi_yoy_pct, core_cpi_yoy_pct,
               breakeven_5y, t10y2y, fedfunds, vix, hy_oas_pct,
               score_velocity, risk_app_velocity, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                snap_date,
                ms.get("overall"),
                overall_v2,
                ms.get("growth"),
                ms.get("inflation"),
                ms.get("fed_policy"),
                ms.get("risk_appetite"),
                _macro_regime(ms.get("overall")),
                ind.get("cfnai_3m"),
                ind.get("payrolls_3m_avg_k"),
                ind.get("cpi_yoy_pct"),
                ind.get("core_cpi_yoy_pct"),
                ind.get("breakeven_5y"),
                ind.get("t10y2y"),
                ind.get("fedfunds"),
                ind.get("vix"),
                ind.get("hy_oas_pct"),
                score_velocity,
                risk_app_velocity,
            ),
        )

        conn.commit()

    finally:
        conn.close()


def set_state(key: str, value: str, db_path: Optional[str] = None) -> None:
    conn = get_db(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO macro_state (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_state(key: str, default: Optional[str] = None, db_path: Optional[str] = None) -> Optional[str]:
    conn = get_db(db_path)
    try:
        row = conn.execute("SELECT value FROM macro_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


def upsert_events(events: List[Dict], db_path: Optional[str] = None) -> None:
    """Upsert macro event rows (days_out and scaling_factor updated daily)."""
    conn = get_db(db_path)
    try:
        for ev in events:
            conn.execute(
                """
                INSERT OR REPLACE INTO macro_events
                  (event_date, event_type, description, days_out, scaling_factor, is_emergency)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ev["event_date"],
                    ev["event_type"],
                    ev.get("description", ""),
                    ev.get("days_out"),
                    ev.get("scaling_factor"),
                    1 if ev.get("is_emergency") else 0,
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Integration API — called by scanner / backtester
# ─────────────────────────────────────────────────────────────────────────────

def get_current_macro_score(
    db_path: Optional[str] = None,
    max_staleness_days: int = 10,
) -> float:
    """
    Return the macro overall score (0–100) from the most recent snapshot.
    Returns 50.0 (neutral) if no data is available.
    Logs a WARNING if the most recent snapshot is older than max_staleness_days.
    """
    conn = get_db(db_path)
    try:
        row = conn.execute(
            "SELECT date, overall FROM macro_score ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if not row or row["overall"] is None:
            return NEUTRAL_SCORE_FALLBACK
        # Staleness detection
        try:
            snapshot_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            days_old = (date.today() - snapshot_date).days
            if days_old > max_staleness_days:
                logger.warning(
                    "COMPASS macro score is %d days stale (last snapshot: %s) — "
                    "weekly job may not have run",
                    days_old, row["date"],
                )
        except Exception:
            pass  # date parsing failure is non-fatal
        return float(row["overall"])
    finally:
        conn.close()


def get_sector_rankings(db_path: Optional[str] = None) -> List[Dict]:
    """
    Return sector RS rankings from the most recent snapshot, sorted by rank_3m.
    Each dict: {ticker, name, category, rs_3m, rs_12m, rank_3m, rank_12m, rrg_quadrant}
    """
    conn = get_db(db_path)
    try:
        latest_date = conn.execute(
            "SELECT MAX(date) AS d FROM sector_rs"
        ).fetchone()["d"]
        if not latest_date:
            return []
        rows = conn.execute(
            """
            SELECT ticker, name, category, rs_3m, rs_12m, rank_3m, rank_12m, rrg_quadrant
            FROM sector_rs
            WHERE date = ?
            ORDER BY rank_3m ASC
            """,
            (latest_date,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_event_scaling_factor(db_path: Optional[str] = None) -> float:
    """
    Return the current position-size scaling factor based on upcoming macro events.
    Value is stored in macro_state by the daily event gate job.
    Returns 1.0 (no scaling) if no event data available.
    """
    val = get_state("event_scaling_factor", default="1.0", db_path=db_path)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 1.0


def get_eligible_underlyings(
    regime: str = "NEUTRAL", db_path: Optional[str] = None
) -> List[str]:
    """
    Return tickers eligible as credit spread underlyings given current macro state.

    Logic:
    - Base universe is always included: SPY, QQQ, IWM
    - In BULL or NEUTRAL regime: add sectors ranked in top 4 by 3M RS
      (restricted to liquid-options sectors)
    - In BEAR regime: add sectors ranked in bottom 4 (for bear call spreads)
    - In BEAR_MACRO (macro score < 45): contract to base universe only
    """
    macro_score = get_current_macro_score(db_path)
    if macro_score < MACRO_BEAR_THRESHOLD:
        # Macro veto: bear macro conditions, contract universe
        return BASE_UNIVERSE.copy()

    rankings = get_sector_rankings(db_path)
    eligible = BASE_UNIVERSE.copy()

    regime_upper = regime.upper()
    for item in rankings:
        ticker = item["ticker"]
        if ticker not in LIQUID_SECTOR_ETFS:
            continue
        rank = item.get("rank_3m") or 99
        if regime_upper in ("BULL", "NEUTRAL") and rank <= SECTOR_RANK_CUTOFF:
            eligible.append(ticker)
        elif regime_upper == "BEAR" and rank >= len(rankings) - (SECTOR_RANK_CUTOFF - 1):
            eligible.append(ticker)

    return list(dict.fromkeys(eligible))  # dedupe, preserve order


def get_latest_snapshot_date(db_path: Optional[str] = None) -> Optional[str]:
    """Return the date string of the most recent snapshot, or None."""
    conn = get_db(db_path)
    try:
        row = conn.execute("SELECT MAX(date) AS d FROM snapshots").fetchone()
        return row["d"] if row else None
    finally:
        conn.close()


def get_snapshot_count(db_path: Optional[str] = None) -> int:
    conn = get_db(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()
        return row["n"] if row else 0
    finally:
        conn.close()


def backfill_macro_score_velocities(db_path: Optional[str] = None) -> int:
    """
    Backfill overall_v2, score_velocity, risk_app_velocity, and updated_at
    for all rows in macro_score (overwrites existing values).

    - score_velocity     = current overall - previous week overall
    - risk_app_velocity  = current risk_appetite - previous week risk_appetite
    - overall_v2         = overall (copy; reserved for alternative formula)
    - updated_at         = datetime('now')

    Rows are processed in chronological order; the first row gets NULL velocities
    (no prior week available). Returns the number of rows updated.
    """
    conn = get_db(db_path)
    try:
        rows = conn.execute(
            "SELECT date, overall, risk_appetite FROM macro_score ORDER BY date ASC"
        ).fetchall()

        updated = 0
        prev_overall: Optional[float] = None
        prev_risk_app: Optional[float] = None

        for row in rows:
            overall = row["overall"]
            risk_app = row["risk_appetite"]

            # First row has no prior week → 0.0 (cleaner than NULL for consumers)
            score_vel: float = 0.0
            risk_vel: float = 0.0

            if prev_overall is not None and overall is not None:
                score_vel = round(float(overall) - float(prev_overall), 2)
            if prev_risk_app is not None and risk_app is not None:
                risk_vel = round(float(risk_app) - float(prev_risk_app), 2)

            conn.execute(
                """
                UPDATE macro_score
                SET overall_v2 = ?, score_velocity = ?, risk_app_velocity = ?,
                    updated_at = datetime('now')
                WHERE date = ?
                """,
                (overall, score_vel, risk_vel, row["date"]),
            )
            updated += 1

            prev_overall = overall
            prev_risk_app = risk_app

        conn.commit()
        logger.info(
            "backfill_macro_score_velocities: updated %d rows in macro_score", updated
        )
        return updated
    finally:
        conn.close()
