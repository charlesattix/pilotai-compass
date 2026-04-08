"""
EXP-1880 — Integrated FOMC + Put/Call Entry Overlays for EXP-1220
==================================================================

Production wiring of the two Wave-2 overlays into one configurable
entry-filter object that the credit-spread strategy can plug in:

  * EXP-1740: FOMC hawkish/dovish HD-score (federalreserve.gov minutes)
              + VIX term-structure slope (^VIX vs ^VIX3M).
  * EXP-1750: SPY put/call volume ratio (IronVault) + VIX inversion
              + put-volume z-score spike + size modifier on PCR fear.

Public API:

    overlay = IntegratedEntryOverlay.from_config(config)
    decision = overlay.allow_entry(date)        # → EntryDecision
    if decision.allow:
        contracts = base_contracts * decision.size_mult

The overlay is *additive* — every filter is independently switchable
via config so the strategy can A/B individual filters in production
without touching code.

REAL DATA ONLY (Rule Zero):
  - FOMC minutes : data/fomc/fomcminutes*.txt  (federalreserve.gov)
  - SPY P/C      : IronVault options_cache.db (Polygon)
  - VIX series   : Yahoo ^VIX / ^VIX3M / ^VIX9D
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1740_sentiment_filter import parse_fomc_minutes
from compass.exp1750_putcall_overlay import (
    load_spy_pc_ratio,
    load_vix_term_structure,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OverlayConfig:
    """Per-filter switches and thresholds. All defaults are the values
    that achieved the documented Wave-2 results."""

    # FOMC sentiment filter (EXP-1740)
    use_fomc: bool = True
    fomc_hawkish_thresh: float = 0.30        # block when HD score ≥ this …
    fomc_block_calendar_days: int = 7        # … within this calendar window after release

    # VIX term structure slope filter (EXP-1740)
    use_vix_slope: bool = True
    vix_slope_min: float = 0.0               # require ^VIX3M − ^VIX ≥ min (contango)

    # Put/Call ratio filter (EXP-1750)
    use_pcr: bool = True
    pcr_low_pct: float = 0.25                # block bottom 25% of rolling pct rank
    pcr_high_pct: float = 0.75
    pcr_lookback: int = 60
    pcr_size_high: float = 1.30              # size multiplier in fear regime
    pcr_size_low: float = 0.50               # size multiplier in residual low (if not blocked)

    # Vol-stress filters (EXP-1750)
    use_vix_inversion: bool = True           # block when ^VIX > ^VIX3M
    use_put_zspike: bool = True              # block when put_zscore_20d > 2

    # Date range to pre-compute features for
    start: str = "2019-06-01"
    end: str = "2026-07-01"

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "OverlayConfig":
        if not d:
            return cls()
        kwargs = {k: d[k] for k in d.keys() if k in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class EntryDecision:
    """Result of a single date lookup."""

    allow: bool
    size_mult: float = 1.0
    blocked_by: tuple = ()         # tuple of filter names that vetoed
    notes: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Implementation
# ─────────────────────────────────────────────────────────────────────────────
class IntegratedEntryOverlay:
    """Pre-computes all features once; per-date lookup is O(1)."""

    def __init__(self, config: OverlayConfig, panel: pd.DataFrame):
        self.config = config
        self.panel = panel  # DatetimeIndex; columns documented below

    # ── factories ──────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config: OverlayConfig | Dict[str, Any] | None = None,
                    *, hd: Any = None) -> "IntegratedEntryOverlay":
        cfg = config if isinstance(config, OverlayConfig) else OverlayConfig.from_dict(config)
        panel = _build_panel(cfg, hd=hd)
        return cls(cfg, panel)

    # ── primary API ────────────────────────────────────────────────────────
    def allow_entry(self, date: datetime | str) -> EntryDecision:
        """Single-date filter decision. Missing data → permissive (allow)."""
        ts = pd.Timestamp(date).normalize()
        if ts not in self.panel.index:
            # use the most recent prior row, if any
            idx = self.panel.index.searchsorted(ts) - 1
            if idx < 0:
                return EntryDecision(allow=True, notes={"reason": "no_panel_data"})
            row = self.panel.iloc[idx]
        else:
            row = self.panel.loc[ts]

        cfg = self.config
        blocked = []

        # 1. FOMC hawkish window
        if cfg.use_fomc:
            hd = row.get("fomc_hd")
            days_since = row.get("fomc_days_since")
            if pd.notna(hd) and pd.notna(days_since):
                if hd >= cfg.fomc_hawkish_thresh and days_since <= cfg.fomc_block_calendar_days:
                    blocked.append("fomc_hawkish")

        # 2. VIX term-structure slope
        if cfg.use_vix_slope:
            slope = row.get("vix_slope")
            if pd.notna(slope) and slope < cfg.vix_slope_min:
                blocked.append("vix_slope")

        # 3. PCR low (complacency)
        size_mult = 1.0
        if cfg.use_pcr:
            pcr_rank = row.get("pcr_pct_rank")
            if pd.notna(pcr_rank):
                if pcr_rank < cfg.pcr_low_pct:
                    blocked.append("pcr_complacent")
                elif pcr_rank >= cfg.pcr_high_pct:
                    size_mult = cfg.pcr_size_high

        # 4. VIX inversion
        if cfg.use_vix_inversion:
            inv = row.get("vix_inverted")
            if pd.notna(inv) and bool(inv):
                blocked.append("vix_inversion")

        # 5. Put-volume spike
        if cfg.use_put_zspike:
            pz = row.get("put_zscore_20d")
            if pd.notna(pz) and pz > 2.0:
                blocked.append("put_zspike")

        return EntryDecision(
            allow=(len(blocked) == 0),
            size_mult=size_mult if not blocked else 1.0,
            blocked_by=tuple(blocked),
            notes={
                "fomc_hd": float(row.get("fomc_hd")) if pd.notna(row.get("fomc_hd")) else None,
                "vix_slope": float(row.get("vix_slope")) if pd.notna(row.get("vix_slope")) else None,
                "pcr_pct_rank": float(row.get("pcr_pct_rank")) if pd.notna(row.get("pcr_pct_rank")) else None,
            },
        )

    # ── batch helper for the standalone trade engine ──────────────────────
    def filter_trades(self, trades: list[Dict]) -> list[Dict]:
        """Apply allow_entry() to a list of EXP-1220 trades. Adjusts contract
        count and recomputes PnL proportionally for size_mult ≠ 1."""
        kept = []
        for t in trades:
            d = self.allow_entry(t["entry_date"])
            if not d.allow:
                continue
            if d.size_mult != 1.0:
                base_cts = max(1, t["contracts"])
                new_cts = max(1, int(round(base_cts * d.size_mult)))
                scale = new_cts / base_cts
                t = dict(t)
                t["contracts"] = new_cts
                t["pnl"] = round(t["pnl"] * scale, 2)
                t["overlay_size_mult"] = d.size_mult
            kept.append(t)
        return kept


# ─────────────────────────────────────────────────────────────────────────────
# Panel construction (called once per overlay instance)
# ─────────────────────────────────────────────────────────────────────────────
def _build_panel(cfg: OverlayConfig, *, hd: Any = None) -> pd.DataFrame:
    """Daily panel keyed on a SPY trading-day index. Columns:
       fomc_hd, fomc_days_since, vix, vix3m, vix_slope, vix_inverted,
       pcr, pcr_pct_rank, put_zscore_20d.
    """
    import yfinance as yf
    from shared.iron_vault import IronVault

    spy = yf.download("SPY", start=cfg.start, end=cfg.end, progress=False, auto_adjust=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).normalize()
    panel = pd.DataFrame(index=spy.index)

    # ── FOMC features ──
    fomc = parse_fomc_minutes()
    if fomc:
        fd = pd.DataFrame([f.__dict__ for f in fomc])
        fd["date"] = pd.to_datetime(fd["date"])
        fd = fd.sort_values("date").reset_index(drop=True)
        hd_col = np.full(len(panel), np.nan)
        ds_col = np.full(len(panel), np.nan)
        j = 0
        for i, day in enumerate(panel.index):
            while j + 1 < len(fd) and fd.iloc[j + 1]["date"] <= day:
                j += 1
            if fd.iloc[j]["date"] <= day:
                hd_col[i] = fd.iloc[j]["hd_score"]
                ds_col[i] = (day - fd.iloc[j]["date"]).days
        panel["fomc_hd"] = hd_col
        panel["fomc_days_since"] = ds_col
    else:
        panel["fomc_hd"] = np.nan
        panel["fomc_days_since"] = np.nan

    # ── VIX term structure ──
    vix_df = load_vix_term_structure(cfg.start, cfg.end)
    if not vix_df.empty:
        vix_df.index = pd.to_datetime(vix_df.index).normalize()
        panel["vix"] = vix_df["vix"].reindex(panel.index).ffill()
        panel["vix3m"] = vix_df["vix3m"].reindex(panel.index).ffill()
        panel["vix_slope"] = panel["vix3m"] - panel["vix"]
        panel["vix_inverted"] = (panel["vix"] > panel["vix3m"]).astype(int)
    else:
        panel["vix"] = panel["vix3m"] = panel["vix_slope"] = np.nan
        panel["vix_inverted"] = 0

    # ── SPY put/call ratio ──
    if hd is None:
        hd = IronVault.instance()
    pcr_df = load_spy_pc_ratio(hd, cfg.start, cfg.end)
    if not pcr_df.empty:
        pcr_df.index = pd.to_datetime(pcr_df.index).normalize()
        panel["pcr"] = pcr_df["pcr"].reindex(panel.index).ffill()
        panel["pcr_pct_rank"] = panel["pcr"].rolling(cfg.pcr_lookback).rank(pct=True)
        panel["put_zscore_20d"] = pcr_df["put_zscore_20d"].reindex(panel.index).ffill()
    else:
        panel["pcr"] = panel["pcr_pct_rank"] = panel["put_zscore_20d"] = np.nan

    panel.index.name = "date"
    return panel
