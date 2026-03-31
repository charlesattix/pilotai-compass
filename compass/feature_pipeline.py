"""
Feature Pipeline — stationary, normalized features for ML models.

Fixes identified in the feature audit:
  1. spy_price:   raw price drifts → z-score vs 60-day rolling mean/std
  2. vix_level:   raw VIX drifts by regime → z-score vs 60-day window
  3. contracts:   raw count, unbounded → log(1 + x)
  4. short_strike: raw price → dropped (otm_pct already captures this)
  5. vix_change_5d: absolute points → pct of current VIX
  6. net_credit / spread_width / max_loss_per_unit: dollar-denominated →
     normalized via credit_to_width and loss_to_width ratios
  7. Imputation: 0.0 fill replaced with domain-aware defaults

All transforms operate on a raw trade DataFrame (as produced by
collect_training_data.enrich_trades) and return a clean feature matrix
ready for model training or inference.

Usage:
    from compass.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    features_df = pipeline.transform(raw_df)
    # features_df has only stationary, bounded columns
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from shared.indicators import sanitize_features

logger = logging.getLogger(__name__)

# ── Raw features that must NOT appear in the output ──────────────────────
# These are either non-stationary or redundant with a normalized version.
_RAW_PRICE_COLUMNS = frozenset({
    "spy_price",
    "short_strike",
})

# ── Domain-aware imputation defaults ─────────────────────────────────────
# Key: column name, Value: fill value when missing.
# Rationale in comments.
_IMPUTATION_DEFAULTS: Dict[str, float] = {
    # 0.0 means "no spread between IV and RV" — neutral, not bullish
    "vol_premium": 0.0,
    "vol_premium_pct": 0.0,
    # Percentile 50 = median; more honest than 0 which means "extreme low"
    "iv_rank": 50.0,
    "iv_percentile": 50.0,
    "vix_percentile_20d": 50.0,
    "vix_percentile_50d": 50.0,
    "vix_percentile_100d": 50.0,
    # RSI 50 = neutral momentum
    "rsi_14": 50.0,
    # No momentum = 0 return (correct default)
    "momentum_5d_pct": 0.0,
    "momentum_10d_pct": 0.0,
    # "Unknown" event distance: use 30 (about a month, below the 999 sentinel)
    "days_to_earnings": 30.0,
    "days_to_fomc": 30.0,
    "days_to_cpi": 15.0,
    # Neutral event risk
    "event_risk_score": 0.2,
    # Vol: use 20% annualized as "normal" rather than 0
    "realized_vol_atr20": 20.0,
    "realized_vol_5d": 20.0,
    "realized_vol_10d": 20.0,
    "realized_vol_20d": 20.0,
}

# Columns where 0.0 fill is correct (returns, distances, slopes, binary flags)
_ZERO_FILL_COLUMNS = frozenset({
    "return_5d", "return_10d", "return_20d",
    "spy_return_5d", "spy_return_20d",
    "dist_from_ma20_pct", "dist_from_ma50_pct",
    "dist_from_ma80_pct", "dist_from_ma200_pct",
    "dist_from_sma20_pct", "dist_from_sma50_pct", "dist_from_sma200_pct",
    "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "vix_change_1d", "vix_change_5d",
    "macd", "macd_signal", "macd_histogram",
    "days_since_last_trade",
    "is_opex_week", "is_monday", "is_month_end",
    "rsi_oversold", "rsi_overbought",
    "iv_rank_high", "iv_rank_low",
})

# ── Z-score rolling window size ─────────────────────────────────────────
ZSCORE_WINDOW = 60  # trading days (~3 months)


def _zscore_column(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Compute rolling z-score: (x - rolling_mean) / rolling_std.

    Uses an expanding window for the first `window` rows so no values
    are dropped.  Clips to [-4, 4] to prevent extreme outliers.
    """
    roll_mean = series.expanding(min_periods=1).mean()
    roll_std = series.expanding(min_periods=1).std()

    # Once we have enough data, switch to fixed rolling window
    if len(series) > window:
        roll_mean_fixed = series.rolling(window, min_periods=max(10, window // 3)).mean()
        roll_std_fixed = series.rolling(window, min_periods=max(10, window // 3)).std()
        # Use fixed where available, expanding as fallback
        mask = roll_mean_fixed.notna()
        roll_mean = roll_mean.where(~mask, roll_mean_fixed)
        roll_std = roll_std.where(~mask, roll_std_fixed)

    # Avoid division by zero
    roll_std = roll_std.replace(0, np.nan)
    z = (series - roll_mean) / roll_std
    z = z.fillna(0.0)
    return z.clip(-4.0, 4.0)


def _log1p_column(series: pd.Series) -> pd.Series:
    """Apply log(1 + |x|) * sign(x) transform for unbounded positive counts."""
    return np.sign(series) * np.log1p(series.abs())


def _credit_to_width(row: pd.Series) -> float:
    """Compute credit / spread_width ratio. Returns 0.0 if width is zero/missing."""
    sw = row.get("spread_width", 0)
    nc = row.get("net_credit", 0)
    if sw is None or nc is None or pd.isna(sw) or pd.isna(nc) or sw <= 0:
        return 0.0
    return nc / sw


def _loss_to_width(row: pd.Series) -> float:
    """Compute max_loss_per_unit / spread_width ratio."""
    sw = row.get("spread_width", 0)
    ml = row.get("max_loss_per_unit", 0)
    if sw is None or ml is None or pd.isna(sw) or pd.isna(ml) or sw <= 0:
        return 0.0
    return ml / sw


class FeaturePipeline:
    """Transform raw trade data into stationary, normalized features.

    Steps applied:
      1. Drop raw price columns (spy_price, short_strike).
      2. Z-score normalize VIX and SPY-derived price levels.
      3. Log-transform contract counts.
      4. Replace dollar-denominated trade features with ratios.
      5. Cap days_to_* sentinel values (999 → 30).
      6. Apply domain-aware imputation for missing values.
      7. One-hot encode categoricals.
      8. Sanitize inf/NaN as a final safety net.

    The pipeline is *stateless* — it does not fit/store parameters from
    training data (no train/test leakage risk). All normalizations use
    only the data available in each row or rolling lookback.
    """

    def __init__(
        self,
        numeric_features: Optional[Sequence[str]] = None,
        categorical_features: Optional[Sequence[str]] = None,
        zscore_window: int = ZSCORE_WINDOW,
        pruned: bool = True,
    ):
        """
        Args:
            numeric_features: Override numeric feature list. If None, uses defaults.
            categorical_features: Override categorical features. Defaults to
                regime, strategy_type, spread_type.
            zscore_window: Rolling window for z-score normalization.
            pruned: When True (default), drop the 10 features identified as
                harmful or noise by the ablation analysis. Set to False for
                backward-compatible full 31-feature output.
        """
        self.numeric_features = list(numeric_features) if numeric_features else None
        self.categorical_features = (
            list(categorical_features) if categorical_features is not None
            else ["regime", "strategy_type", "spread_type"]
        )
        self.zscore_window = zscore_window
        self.pruned = pruned

    # ── Default numeric feature list ─────────────────────────────────────

    @staticmethod
    def default_numeric_features() -> List[str]:
        """Return the canonical list of numeric features after pipeline transforms.

        This replaces NUMERIC_FEATURES from walk_forward.py with cleaned versions:
          - spy_price → spy_price_zscore
          - vix (raw) → vix_zscore
          - contracts → contracts_log
          - net_credit, spread_width, max_loss_per_unit → credit_to_width, loss_to_width
          - short_strike, otm_pct → dropped (otm_pct was in "harmful" list)
          - vix_change_5d → vix_change_5d_pct
        """
        return [
            # Calendar / timing (bounded, stationary)
            "days_since_last_trade",

            # Momentum & trend (returns-based, stationary)
            "rsi_14",
            "momentum_5d_pct",
            "momentum_10d_pct",

            # Volatility context (z-scored or percentile-based)
            "vix_zscore",
            "vix_change_5d_pct",
            "vix_percentile_50d",
            "vix_percentile_100d",
            "iv_rank",

            # Price structure (all %-based distances, stationary)
            "spy_price_zscore",
            "dist_from_ma20_pct",
            "dist_from_ma50_pct",
            "dist_from_ma80_pct",
            "dist_from_ma200_pct",
            "ma50_slope_ann_pct",

            # Realized volatility (annualized %, stationary)
            "realized_vol_atr20",
            "realized_vol_20d",

            # Trade structure (ratios, bounded 0-1ish)
            "credit_to_width",
            "loss_to_width",
            "contracts_log",
        ]

    # ── Main transform ───────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a raw trade DataFrame into a clean feature matrix.

        Args:
            df: Raw DataFrame from collect_training_data.enrich_trades()
                or equivalent.  Must have columns like spy_price, vix,
                net_credit, spread_width, etc.

        Returns:
            DataFrame with deterministic column order, all values finite.
        """
        out = df.copy()

        # Step 1: Z-score raw prices
        if "spy_price" in out.columns:
            out["spy_price_zscore"] = _zscore_column(
                out["spy_price"].astype(float), self.zscore_window,
            )
        else:
            out["spy_price_zscore"] = 0.0

        if "vix" in out.columns:
            out["vix_zscore"] = _zscore_column(
                out["vix"].astype(float), self.zscore_window,
            )
        else:
            out["vix_zscore"] = 0.0

        # Step 2: Normalize VIX change from absolute points to %
        if "vix" in out.columns and "vix_change_5d" not in out.columns:
            out["vix_change_5d_pct"] = 0.0
        elif "vix_change_5d" in out.columns and "vix" in out.columns:
            vix = out["vix"].replace(0, np.nan)
            out["vix_change_5d_pct"] = (out["vix_change_5d"] / vix * 100).fillna(0.0)
        else:
            out["vix_change_5d_pct"] = 0.0

        # Step 3: Log-transform contracts
        if "contracts" in out.columns:
            out["contracts_log"] = _log1p_column(out["contracts"].fillna(0).astype(float))
        else:
            out["contracts_log"] = 0.0

        # Step 4: Trade structure ratios (replace dollar features)
        out["credit_to_width"] = out.apply(_credit_to_width, axis=1)
        out["loss_to_width"] = out.apply(_loss_to_width, axis=1)

        # Step 5: Cap sentinel values
        for col in ["days_to_earnings", "days_to_fomc", "days_to_cpi"]:
            if col in out.columns:
                out[col] = out[col].clip(upper=60)

        # Step 6: Select and order features
        numeric_cols = self.numeric_features or self.default_numeric_features()

        parts: list[pd.DataFrame] = []

        # Numeric: fill missing with domain-aware defaults
        present_numeric = [c for c in numeric_cols if c in out.columns]
        missing_numeric = [c for c in numeric_cols if c not in out.columns]
        if missing_numeric:
            logger.debug("Pipeline: %d numeric features not in data, filling defaults: %s",
                         len(missing_numeric), missing_numeric)

        num_df = pd.DataFrame(index=out.index)
        for col in numeric_cols:
            if col in out.columns:
                default = _IMPUTATION_DEFAULTS.get(col, 0.0)
                num_df[col] = out[col].fillna(default).astype(float)
            else:
                default = _IMPUTATION_DEFAULTS.get(col, 0.0)
                num_df[col] = default
        parts.append(num_df)

        # Categorical: one-hot encode
        for col in self.categorical_features:
            if col in out.columns:
                dummies = pd.get_dummies(out[col], prefix=col, dummy_na=False)
                parts.append(dummies)

        result = pd.concat(parts, axis=1)
        clean = sanitize_features(result.values.astype(np.float64))
        result = pd.DataFrame(clean, columns=result.columns, index=result.index)

        # Step 8: Prune harmful/noise features (post-ablation)
        if self.pruned:
            from compass.features import PRUNED_REMOVED
            to_drop = [c for c in PRUNED_REMOVED if c in result.columns]
            if to_drop:
                result = result.drop(columns=to_drop)

        return result

    # ── Convenience: get feature names after transform ───────────────────

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Return the column names that transform() would produce for df."""
        return list(self.transform(df.head(1)).columns)
