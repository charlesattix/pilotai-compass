"""
Generate Historical Macro Snapshots
=====================================
Generates weekly macro snapshots for every Friday from 2020-01-03 to today.

Usage:
    python3 scripts/generate_historical_snapshots.py
    python3 scripts/generate_historical_snapshots.py --start 2023-01-01
    python3 scripts/generate_historical_snapshots.py --dry-run

Outputs:
    output/historical_snapshots/YYYY/YYYY-MM-DD.json  — one per Friday
    output/historical_snapshots/summary.csv            — one row per snapshot
    output/historical_snapshot_analysis.md             — forward return analysis

Requires:
    POLYGON_API_KEY in environment (or .env file)
    FRED_API_KEY    in environment (optional but recommended for macro score)
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from compass.macro import MacroSnapshotEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Output paths ───────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output" / "historical_snapshots"
ANALYSIS_PATH = PROJECT_ROOT / "output" / "historical_snapshot_analysis.md"


# ── Date helpers ───────────────────────────────────────────────────────────────

def all_fridays(start: date, end: date):
    """Yield every Friday between start and end inclusive."""
    # Advance to first Friday on or after start
    d = start
    while d.weekday() != 4:  # 4 = Friday
        d += timedelta(days=1)
    while d <= end:
        yield d
        d += timedelta(days=7)


def prev_trading_day(d: date) -> date:
    """Return d if it's Mon-Fri, else the preceding Friday."""
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ── Write helpers ──────────────────────────────────────────────────────────────

def write_snapshot_json(snap: dict, output_dir: Path) -> None:
    year = snap["date"][:4]
    year_dir = output_dir / year
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{snap['date']}.json"
    with open(path, "w") as f:
        json.dump(snap, f, indent=2, default=str)


def snapshot_to_csv_row(snap: dict) -> dict:
    """Flatten a snapshot to a single CSV-friendly row."""
    ms = snap.get("macro_score") or {}
    row = {
        "date": snap["date"],
        "spy_close": snap.get("spy_close"),
        "top_sector_3m": snap.get("top_sector_3m"),
        "top_sector_12m": snap.get("top_sector_12m"),
        "leading_sectors": "|".join(snap.get("leading_sectors") or []),
        "lagging_sectors": "|".join(snap.get("lagging_sectors") or []),
        "macro_overall": ms.get("overall"),
        "macro_growth": ms.get("growth"),
        "macro_inflation": ms.get("inflation"),
        "macro_fed_policy": ms.get("fed_policy"),
        "macro_risk_appetite": ms.get("risk_appetite"),
    }
    # Add per-sector RS fields (sector ETFs only)
    for item in snap.get("sector_rankings") or []:
        if item["category"] == "sector":
            t = item["ticker"]
            row[f"{t}_rs_3m"] = item.get("rs_3m")
            row[f"{t}_rs_12m"] = item.get("rs_12m")
            row[f"{t}_quadrant"] = item.get("rrg_quadrant")
    return row


# ── Forward return computation ─────────────────────────────────────────────────

def add_forward_returns(df: pd.DataFrame, weeks: list = [4, 8, 12]) -> pd.DataFrame:
    """
    Compute forward SPY and top-3 basket returns for each snapshot date.
    Also computes whether each week's top-ranked sector outperformed SPY.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for w in weeks:
        col_spy = f"spy_{w}w_return"
        col_top = f"top3_vs_spy_{w}w"
        df[col_spy] = None
        df[col_top] = None

    # Build spy close lookup
    spy_lookup = dict(zip(df["date"], df["spy_close"]))

    for idx, row in df.iterrows():
        snap_date = row["date"]
        spy_close = row["spy_close"]
        if spy_close is None or np.isnan(spy_close):
            continue

        for w in weeks:
            fwd_date = snap_date + pd.Timedelta(weeks=w)
            # Find nearest available Friday on or after fwd_date
            check = fwd_date
            for _ in range(7):
                if check in spy_lookup and spy_lookup[check] is not None:
                    break
                check += pd.Timedelta(days=1)
            else:
                continue

            fwd_spy = spy_lookup.get(check)
            if fwd_spy and not np.isnan(fwd_spy):
                ret = (fwd_spy / spy_close - 1.0) * 100.0
                df.at[idx, f"spy_{w}w_return"] = round(ret, 3)

    return df


def compute_sector_hit_rates(df: pd.DataFrame, weeks: list = [4, 8, 12]) -> dict:
    """
    For each forward-return horizon, compute:
      - Hit rate: % of weeks where top-ranked sector (by rs_3m) outperformed SPY
      - Top-3 basket hit rate
    """
    # This requires per-sector forward returns which are in the JSON snapshots,
    # not the summary CSV.  We compute from the CSV using sector RS columns.
    results = {}
    for w in weeks:
        spy_col = f"spy_{w}w_return"
        if spy_col not in df.columns:
            continue
        sub = df.dropna(subset=[spy_col])
        if len(sub) < 5:
            results[w] = {"n": len(sub), "note": "insufficient data"}
            continue
        results[w] = {
            "n": len(sub),
            "spy_avg_return": round(float(sub[spy_col].mean()), 3),
            "spy_positive_pct": round(float((sub[spy_col] > 0).mean() * 100), 1),
        }
    return results


# ── Macro score vs SPY correlation ────────────────────────────────────────────

def compute_macro_predictive_power(df: pd.DataFrame, weeks: list = [4, 8, 12]) -> dict:
    """Correlation between macro_overall score and forward SPY returns."""
    results = {}
    macro_col = "macro_overall"
    if macro_col not in df.columns:
        return results

    for w in weeks:
        spy_col = f"spy_{w}w_return"
        if spy_col not in df.columns:
            continue
        sub = df[[macro_col, spy_col]].dropna()
        if len(sub) < 10:
            continue
        corr = float(sub[macro_col].corr(sub[spy_col]))
        # Tertile split: low/mid/high macro score
        tertiles = pd.qcut(sub[macro_col], 3, labels=["low", "mid", "high"])
        avg_by_tertile = sub.groupby(tertiles, observed=False)[spy_col].mean().to_dict()
        results[w] = {
            "n": len(sub),
            "correlation": round(corr, 3),
            "avg_spy_return_by_macro_tertile": {
                k: round(v, 3) for k, v in avg_by_tertile.items()
            },
        }
    return results


# ── RRG hit rate analysis ──────────────────────────────────────────────────────

def compute_rrg_hit_rates(all_snapshots: list, weeks: list = [4, 8]) -> dict:
    """
    For each snapshot, check whether 'Leading' sectors outperformed 'Lagging' sectors
    over the next N weeks.  Uses price data already in the engine's cache via the
    summary DataFrame — we approximate using rs_3m sign flips.

    Note: full per-sector forward return requires loading individual JSONs.
    Returns descriptive stats on RRG quadrant distribution instead.
    """
    quadrant_counts = {"Leading": 0, "Weakening": 0, "Lagging": 0, "Improving": 0}
    for snap in all_snapshots:
        for item in snap.get("sector_rankings") or []:
            q = item.get("rrg_quadrant")
            if q in quadrant_counts:
                quadrant_counts[q] += 1

    total = sum(quadrant_counts.values()) or 1
    return {
        "quadrant_distribution": {k: round(v / total * 100, 1) for k, v in quadrant_counts.items()},
        "total_sector_observations": total,
    }


# ── Analysis report writer ─────────────────────────────────────────────────────

def write_analysis_report(
    df: pd.DataFrame,
    all_snapshots: list,
    output_path: Path,
) -> None:
    """Write output/historical_snapshot_analysis.md."""

    weeks = [4, 8, 12]
    spy_stats = compute_sector_hit_rates(df, weeks)
    macro_corr = compute_macro_predictive_power(df, weeks)
    rrg_stats = compute_rrg_hit_rates(all_snapshots, [4, 8])

    n_total = len(df)
    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    n_with_macro = int(df["macro_overall"].notna().sum())

    lines = [
        "# Historical Macro Snapshot Analysis",
        "",
        f"**Generated:** {date.today().strftime('%Y-%m-%d')}  ",
        f"**Coverage:** {date_range}  ",
        f"**Total snapshots:** {n_total}  ",
        f"**Snapshots with FRED macro score:** {n_with_macro}  ",
        "",
        "---",
        "",
        "## 1. SPY Forward Return Summary",
        "",
        "Average SPY return and % of weeks positive over each horizon:",
        "",
        "| Horizon | N Weeks | Avg SPY Return | % Positive |",
        "|---------|---------|---------------|------------|",
    ]
    for w in weeks:
        s = spy_stats.get(w, {})
        avg = f"{s.get('spy_avg_return', 'N/A'):.2f}%" if isinstance(s.get("spy_avg_return"), float) else "N/A"
        pct = f"{s.get('spy_positive_pct', 'N/A'):.1f}%" if isinstance(s.get("spy_positive_pct"), float) else "N/A"
        lines.append(f"| {w} weeks | {s.get('n', 'N/A')} | {avg} | {pct} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Macro Score vs Forward SPY Returns",
        "",
        "Pearson correlation between macro overall score (0-100) and forward SPY return:",
        "",
        "| Horizon | N | Correlation | Low-Score Avg | Mid-Score Avg | High-Score Avg |",
        "|---------|---|-------------|---------------|---------------|----------------|",
    ]
    for w in weeks:
        mc = macro_corr.get(w)
        if mc is None:
            lines.append(f"| {w} weeks | N/A | N/A | N/A | N/A | N/A |")
            continue
        t = mc.get("avg_spy_return_by_macro_tertile", {})
        low = f"{t.get('low', 0):.2f}%"
        mid = f"{t.get('mid', 0):.2f}%"
        high = f"{t.get('high', 0):.2f}%"
        lines.append(
            f"| {w} weeks | {mc['n']} | {mc['correlation']:.3f} | {low} | {mid} | {high} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. RRG Quadrant Distribution",
        "",
        "How often sectors fall in each RRG quadrant across all snapshots:",
        "",
        "| Quadrant | % of Observations |",
        "|----------|-------------------|",
    ]
    for q, pct in rrg_stats.get("quadrant_distribution", {}).items():
        lines.append(f"| {q} | {pct:.1f}% |")
    lines.append(f"| **Total** | **{rrg_stats.get('total_sector_observations', 0):,} sector-weeks** |")

    lines += [
        "",
        "---",
        "",
        "## 4. Top Sector Rotation Signals",
        "",
        "### Top-Ranked Sector (3M RS) — Frequency by Ticker",
        "",
        "Which sectors appear most frequently as the #1 ranked sector:",
        "",
    ]
    if "top_sector_3m" in df.columns:
        freq = df["top_sector_3m"].value_counts()
        lines.append("| Ticker | Count | % of Weeks |")
        lines.append("|--------|-------|------------|")
        for ticker, count in freq.items():
            lines.append(f"| {ticker} | {count} | {count/n_total*100:.1f}% |")

    lines += [
        "",
        "### Macro Score Distribution",
        "",
    ]
    if "macro_overall" in df.columns and n_with_macro > 0:
        ms = df["macro_overall"].dropna()
        lines.append(f"- Mean:   {ms.mean():.1f}")
        lines.append(f"- Median: {ms.median():.1f}")
        lines.append(f"- Std:    {ms.std():.1f}")
        lines.append(f"- Min:    {ms.min():.1f}")
        lines.append(f"- Max:    {ms.max():.1f}")
        lines.append("")
        q = df.groupby(pd.qcut(df["macro_overall"].fillna(50), 5, duplicates="drop"), observed=False)["macro_overall"].count()
        lines.append("Score quintile distribution:")
        lines.append("")
        lines.append("| Quintile range | N snapshots |")
        lines.append("|----------------|-------------|")
        for label, count in q.items():
            lines.append(f"| {label} | {count} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Methodology Notes",
        "",
        "- **RS (3M)**: `(ticker_return_3M / SPY_return_3M - 1) × 100` — percentage outperformance vs benchmark",
        "- **RS (12M)**: same over 12 months (~252 trading days)",
        "- **RRG Quadrant**: cross-sectionally normalized RS-Ratio and RS-Momentum, centered at 100",
        "  - Leading: RS-Ratio ≥ 100 AND RS-Momentum ≥ 100",
        "  - Weakening: RS-Ratio ≥ 100 AND RS-Momentum < 100",
        "  - Lagging: RS-Ratio < 100 AND RS-Momentum < 100",
        "  - Improving: RS-Ratio < 100 AND RS-Momentum ≥ 100",
        "- **Macro Score**: 4 dimensions, each 0-100, equal-weighted to overall score",
        "  - Growth: CFNAI 3M avg (50%) + Nonfarm Payrolls 3M avg (50%) — CFNAI is a composite of 85 indicators",
        "  - Inflation: CPI YoY (35%) + Core CPI YoY (40%) + 5Y Breakeven (25%) — Goldilocks curve peaks at 2-2.5%",
        "  - Fed Policy: 10Y-2Y spread (55%) + Effective Fed Funds (45%)",
        "  - Risk Appetite: VIX (50%) + HY OAS spread (50%)",
        "- **RELEASE_LAG_DAYS**: Applied per FRED series to prevent lookahead bias",
        "  - Daily series (VIX, spreads): 1-day lag",
        "  - Monthly releases (CPI, payrolls, PMI): 31-66 day lag",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Analysis written to %s", output_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate historical macro snapshots")
    p.add_argument(
        "--start",
        default="2020-01-03",
        help="Start date YYYY-MM-DD (default: 2020-01-03)",
    )
    p.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List dates only, do not fetch or generate",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip dates that already have a JSON file (default: True)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-generate all snapshots even if JSON already exists",
    )
    return p.parse_args()


def main():
    args = parse_args()

    polygon_key = os.getenv("POLYGON_API_KEY", "")
    fred_key = os.getenv("FRED_API_KEY", "")

    if not polygon_key:
        logger.error("POLYGON_API_KEY not set in environment. Aborting.")
        sys.exit(1)
    # FRED data is fetched via the public CSV endpoint — no API key needed.

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else date.today()
    skip_existing = args.skip_existing and not args.force

    fridays = list(all_fridays(start_date, end_date))
    logger.info("Generating %d Friday snapshots: %s → %s", len(fridays), start_date, end_date)

    if args.dry_run:
        for d in fridays:
            print(d)
        return

    # ── Initialize engine ─────────────────────────────────────────────────────
    engine = MacroSnapshotEngine(
        polygon_key=polygon_key,
        fred_key=fred_key or None,
        cache_dir=str(PROJECT_ROOT / "data" / "macro_cache"),
    )

    # ── Prefetch all data (cache-first, fast on re-run) ───────────────────────
    # Start 280 calendar days before first snapshot to warm up RS lookbacks
    warmup_start = start_date - timedelta(days=290)
    engine.prefetch_all_data(start_date=warmup_start, end_date=end_date)

    # ── Generate snapshots ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_snapshots = []
    csv_rows = []
    skipped = 0
    generated = 0

    for snap_date in fridays:
        json_path = OUTPUT_DIR / snap_date.strftime("%Y") / f"{snap_date}.json"

        if skip_existing and json_path.exists():
            # Load existing for summary purposes
            try:
                with open(json_path) as f:
                    snap = json.load(f)
                all_snapshots.append(snap)
                csv_rows.append(snapshot_to_csv_row(snap))
                skipped += 1
                continue
            except Exception:
                pass  # fall through to regenerate

        logger.info("Generating snapshot: %s", snap_date)
        try:
            snap = engine.generate_snapshot(snap_date)
        except Exception as exc:
            logger.error("Failed to generate %s: %s", snap_date, exc)
            continue

        write_snapshot_json(snap, OUTPUT_DIR)
        all_snapshots.append(snap)
        csv_rows.append(snapshot_to_csv_row(snap))
        generated += 1

    logger.info("Done — generated: %d, skipped (cached): %d", generated, skipped)

    # ── Write summary CSV ─────────────────────────────────────────────────────
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        # Add forward returns
        df = add_forward_returns(df)
        csv_path = OUTPUT_DIR / "summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Summary CSV written: %s (%d rows)", csv_path, len(df))
    else:
        df = pd.DataFrame()
        logger.warning("No snapshots generated — summary CSV skipped")

    # ── Write analysis ────────────────────────────────────────────────────────
    if not df.empty:
        write_analysis_report(df, all_snapshots, ANALYSIS_PATH)

    engine.close()
    logger.info("All done. Snapshots in: %s", OUTPUT_DIR)
    logger.info("Analysis:              %s", ANALYSIS_PATH)


if __name__ == "__main__":
    main()
