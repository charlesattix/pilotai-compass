# Multi-Asset ML Trading System Architecture

## PilotAI v2.0 — Expansion from Options to Futures, Crypto & Prediction Markets

**Date**: 2026-02-18
**Status**: Architecture Plan
**Author**: PilotAI Engineering

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Analysis](#2-current-system-analysis)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Phase 1: Core Infrastructure Refactoring](#4-phase-1-core-infrastructure-refactoring)
5. [Phase 2: Futures Module (ES, NQ, CL)](#5-phase-2-futures-module)
6. [Phase 3: Crypto Module (BTC, ETH, SOL)](#6-phase-3-crypto-module)
7. [Phase 4: Prediction Markets Module (Polymarket)](#7-phase-4-prediction-markets-module)
8. [ML Model Architectures](#8-ml-model-architectures)
9. [Unified Scoring System](#9-unified-scoring-system)
10. [Portfolio-Level Risk Management](#10-portfolio-level-risk-management)
11. [Data Pipeline Design](#11-data-pipeline-design)
12. [Database Schema Extensions](#12-database-schema-extensions)
13. [Backtesting Framework](#13-backtesting-framework)
14. [Web UI Extensions](#14-web-ui-extensions)
15. [Testing Strategy](#15-testing-strategy)
16. [Deployment Approach](#16-deployment-approach)
17. [Implementation Timeline](#17-implementation-timeline)

---

## 1. Executive Summary

This document architects the expansion of PilotAI from a single-asset-class credit spread system into a multi-asset ML-powered trading platform spanning four asset classes:

| Asset Class | Instruments | Leverage | Market Hours | Edge Source |
|---|---|---|---|---|
| **Options** (existing) | SPY/QQQ/IWM credit spreads | 1x (defined risk) | 9:30-16:00 ET | IV premium harvesting, 87% win rate |
| **Futures** (new) | ES, NQ, CL | 20-50x | 23h/day (Sun-Fri) | Term structure, mean reversion, momentum |
| **Crypto** (new) | BTC, ETH, SOL spot + perps | 1-20x | 24/7 | Funding rate arb, on-chain flow, momentum |
| **Prediction Markets** (new) | Polymarket events | 1x | 24/7 | Probabilistic mispricing, information edge |

**Key Design Principles:**
- Extend, don't rewrite — build on existing patterns (DataProvider protocol, MLPipeline, FeatureEngine, SQLite WAL)
- Each asset class gets its own strategy module, feature engine, and ML model
- Unified scoring layer (0-100) enables cross-asset portfolio optimization
- Production options system runs undisturbed while new modules are built and tested

---

## 2. Current System Analysis

### What We're Building On

```
Existing Architecture (15,637 lines Python + Next.js web layer):

├── strategy/           → DataProvider protocol, spread_strategy, technical_analysis
├── ml/                 → XGBoost signal model, HMM regime detector, feature engine
├── shared/             → SQLite WAL DB, types, constants, circuit breaker
├── alerts/             → Telegram, JSON/CSV/text outputs
├── backtest/           → Historical simulation engine
├── tracker/            → Trade lifecycle, P&L dashboard
├── web/                → Next.js 15 + React 19 dashboard
└── paper_trader.py     → Auto-execution engine
```

### Reusable Components (use as-is or extend)
- `shared/database.py` — SQLite WAL for concurrent Python/Node access → extend schema
- `shared/circuit_breaker.py` — Resilience pattern → reuse for all new APIs
- `shared/provider_protocol.py` — DataProvider interface → generalize to multi-asset
- `ml/regime_detector.py` — HMM market regime → extend to futures/crypto regimes
- `ml/position_sizer.py` — Kelly criterion → extend for leverage + cross-asset correlation
- `ml/feature_engine.py` — Feature pipeline → create asset-class-specific subclasses
- `alerts/` — Alert generation → extend for multi-asset alerts
- `backtest/` — Backtesting engine → generalize for any asset class
- `paper_trader.py` — Auto-execution → generalize with execution adapters

### Components That Need Refactoring
- `ml/ml_pipeline.py` — Currently options-only; needs asset-class routing
- `strategy/spread_strategy.py` — Tightly coupled to options; extract base strategy class
- `shared/types.py` — Options-specific TypedDicts; add multi-asset types
- `config.yaml` — Options-only config; extend with asset class sections

---

## 3. System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WEB DASHBOARD                             │
│  Next.js 15 + React 19 (multi-asset views, unified portfolio)   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ API Routes
┌──────────────────────────────┴──────────────────────────────────┐
│                      ORCHESTRATOR LAYER                          │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Options  │ │ Futures  │ │  Crypto  │ │    Prediction    │   │
│  │ Scanner  │ │ Scanner  │ │ Scanner  │ │ Markets Scanner  │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘   │
│       │             │            │                 │              │
│  ┌────┴─────────────┴────────────┴─────────────────┴──────────┐ │
│  │              UNIFIED ML SCORING ENGINE                      │ │
│  │  ┌───────────┐ ┌───────────┐ ┌────────────┐ ┌──────────┐  │ │
│  │  │  Options  │ │  Futures  │ │   Crypto   │ │ PredMkt  │  │ │
│  │  │  ML Model │ │  ML Model │ │  ML Model  │ │ ML Model │  │ │
│  │  └───────────┘ └───────────┘ └────────────┘ └──────────┘  │ │
│  │                                                             │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         Unified Feature Engine (base + per-asset)     │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │                                                             │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         Cross-Asset Scoring (0-100 normalized)        │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              PORTFOLIO RISK MANAGER                          │ │
│  │  Cross-asset correlation │ Position sizing │ Kill switch     │ │
│  │  Leverage management     │ Drawdown control│ Exposure limits │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Alpaca   │ │ Futures  │ │ Exchange │ │   Polymarket     │   │
│  │ (Options)│ │ Broker   │ │ (Crypto) │ │   (PredMkt)      │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────┐
│                     DATA LAYER                                   │
│  SQLite WAL │ Time-series cache │ Model artifacts │ Config       │
└─────────────────────────────────────────────────────────────────┘
```

### New Directory Structure

```
pilotai-credit-spreads/
├── main.py                          # Extended: multi-asset CLI
├── config.yaml                      # Extended: per-asset-class config
├── orchestrator.py                  # NEW: Multi-asset scan coordinator
│
├── strategy/                        # Existing + new strategies
│   ├── base_strategy.py             # NEW: Abstract base for all strategies
│   ├── spread_strategy.py           # Existing (options)
│   ├── futures_strategy.py          # NEW: Futures mean-rev/momentum/spread
│   ├── crypto_strategy.py           # NEW: Crypto funding arb/momentum/breakout
│   ├── prediction_strategy.py       # NEW: Prediction market mispricing
│   ├── options_analyzer.py          # Existing
│   ├── technical_analysis.py        # Existing (generalized)
│   ├── polygon_provider.py          # Existing
│   ├── alpaca_provider.py           # Existing
│   ├── tradier_provider.py          # Existing
│   └── providers/                   # NEW: Multi-asset data providers
│       ├── __init__.py
│       ├── futures_provider.py      # CME/databento/polygon futures data
│       ├── crypto_provider.py       # Exchange APIs (Binance/Coinbase/Bybit)
│       └── polymarket_provider.py   # Polymarket CLOB API
│
├── ml/                              # Extended ML pipeline
│   ├── ml_pipeline.py               # Extended: asset-class routing
│   ├── signal_model.py              # Existing (options model)
│   ├── regime_detector.py           # Extended: multi-asset regimes
│   ├── iv_analyzer.py               # Existing
│   ├── feature_engine.py            # Refactored: base + subclasses
│   ├── position_sizer.py            # Extended: leverage-aware sizing
│   ├── sentiment_scanner.py         # Existing
│   ├── unified_scorer.py            # NEW: Cross-asset scoring normalization
│   ├── futures_model.py             # NEW: Futures-specific ML
│   ├── crypto_model.py              # NEW: Crypto-specific ML
│   ├── prediction_model.py          # NEW: Prediction market ML
│   ├── ensemble.py                  # NEW: Multi-model ensemble combiner
│   └── models/                      # Extended: per-asset model artifacts
│       ├── signal_model.pkl
│       ├── futures_model.pkl
│       ├── crypto_model.pkl
│       └── prediction_model.pkl
│
├── risk/                            # NEW: Portfolio-level risk
│   ├── __init__.py
│   ├── portfolio_manager.py         # Cross-asset risk aggregation
│   ├── correlation_tracker.py       # Real-time correlation matrix
│   ├── leverage_controller.py       # Per-asset + portfolio leverage limits
│   └── drawdown_monitor.py          # Portfolio-wide drawdown circuit breaker
│
├── execution/                       # NEW: Unified execution layer
│   ├── __init__.py
│   ├── base_executor.py             # Abstract execution interface
│   ├── options_executor.py          # Wraps existing Alpaca integration
│   ├── futures_executor.py          # Futures broker API
│   ├── crypto_executor.py           # Exchange execution
│   └── polymarket_executor.py       # Polymarket order placement
│
├── shared/                          # Extended shared infrastructure
│   ├── types.py                     # Extended: multi-asset TypedDicts
│   ├── database.py                  # Extended: new tables
│   ├── data_cache.py                # Extended: multi-asset caching
│   ├── constants.py                 # Extended: new asset class constants
│   ├── provider_protocol.py         # Extended: generalized protocol
│   ├── asset_class.py               # NEW: AssetClass enum + registry
│   ├── circuit_breaker.py           # Existing
│   ├── metrics.py                   # Extended: per-asset metrics
│   ├── scheduler.py                 # Extended: 24/7 scheduling
│   └── indicators.py                # Existing
│
├── backtest/                        # Extended backtesting
│   ├── backtester.py                # Refactored: asset-class-aware
│   ├── futures_backtester.py        # NEW: Futures-specific backtest logic
│   ├── crypto_backtester.py         # NEW: Crypto-specific (funding, 24/7)
│   ├── prediction_backtester.py     # NEW: Event resolution backtester
│   └── performance_metrics.py       # Extended: leverage-adjusted metrics
│
├── data/                            # Extended data storage
│   ├── pilotai.db                   # Extended schema
│   ├── futures/                     # Futures historical data cache
│   ├── crypto/                      # Crypto historical data cache
│   └── predictions/                 # Prediction market historical data
│
└── web/                             # Extended dashboard
    └── app/
        ├── api/
        │   ├── futures/route.ts     # NEW
        │   ├── crypto/route.ts      # NEW
        │   ├── predictions/route.ts # NEW
        │   └── portfolio/route.ts   # NEW: Cross-asset portfolio view
        ├── futures/page.tsx         # NEW
        ├── crypto/page.tsx          # NEW
        ├── predictions/page.tsx     # NEW
        └── portfolio/page.tsx       # NEW: Unified portfolio dashboard
```

---

## 4. Phase 1: Core Infrastructure Refactoring

**Goal**: Generalize the existing options-specific architecture into a multi-asset framework without breaking the production options system.

### 4.1 Asset Class Registry

```python
# shared/asset_class.py

from enum import Enum
from typing import Dict, Type


class AssetClass(Enum):
    OPTIONS = "options"
    FUTURES = "futures"
    CRYPTO = "crypto"
    PREDICTION_MARKETS = "prediction_markets"


class AssetClassConfig:
    """Per-asset-class configuration and metadata."""

    def __init__(
        self,
        asset_class: AssetClass,
        instruments: list[str],
        market_hours: str,          # "market" | "extended" | "24/7" | "23h"
        max_leverage: float,
        default_leverage: float,
        tick_size: float,
        min_position_usd: float,
        max_position_pct: float,    # max % of portfolio
        scan_interval_minutes: int,
    ):
        self.asset_class = asset_class
        self.instruments = instruments
        self.market_hours = market_hours
        self.max_leverage = max_leverage
        self.default_leverage = default_leverage
        self.tick_size = tick_size
        self.min_position_usd = min_position_usd
        self.max_position_pct = max_position_pct
        self.scan_interval_minutes = scan_interval_minutes


# Default configurations
ASSET_CLASS_CONFIGS: Dict[AssetClass, AssetClassConfig] = {
    AssetClass.OPTIONS: AssetClassConfig(
        asset_class=AssetClass.OPTIONS,
        instruments=["SPY", "QQQ", "IWM"],
        market_hours="market",
        max_leverage=1.0,           # Defined risk spreads
        default_leverage=1.0,
        tick_size=0.01,
        min_position_usd=500,
        max_position_pct=0.10,
        scan_interval_minutes=30,
    ),
    AssetClass.FUTURES: AssetClassConfig(
        asset_class=AssetClass.FUTURES,
        instruments=["ES", "NQ", "CL"],
        market_hours="23h",
        max_leverage=20.0,
        default_leverage=5.0,
        tick_size=0.25,             # ES tick = $12.50 per contract
        min_position_usd=5000,
        max_position_pct=0.15,
        scan_interval_minutes=15,
    ),
    AssetClass.CRYPTO: AssetClassConfig(
        asset_class=AssetClass.CRYPTO,
        instruments=["BTC", "ETH", "SOL"],
        market_hours="24/7",
        max_leverage=10.0,          # Conservative for crypto
        default_leverage=3.0,
        tick_size=0.01,
        min_position_usd=100,
        max_position_pct=0.10,
        scan_interval_minutes=10,
    ),
    AssetClass.PREDICTION_MARKETS: AssetClassConfig(
        asset_class=AssetClass.PREDICTION_MARKETS,
        instruments=[],             # Dynamic — fetched from Polymarket
        market_hours="24/7",
        max_leverage=1.0,           # No leverage on predictions
        default_leverage=1.0,
        tick_size=0.01,
        min_position_usd=50,
        max_position_pct=0.05,
        scan_interval_minutes=60,
    ),
}
```

### 4.2 Base Strategy Protocol

```python
# strategy/base_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from shared.asset_class import AssetClass


class BaseStrategy(ABC):
    """Abstract base class for all asset-class strategies."""

    asset_class: AssetClass

    @abstractmethod
    def scan(self, instruments: List[str], config: Dict) -> List[Dict]:
        """Scan for trading opportunities.

        Returns:
            List of opportunity dicts, each containing at minimum:
            - instrument: str
            - asset_class: AssetClass
            - signal_type: str (e.g. 'bull_put_spread', 'long_momentum', 'funding_arb')
            - entry_price: float
            - target_price: float
            - stop_price: float
            - score: float (0-100, raw strategy score before ML)
            - metadata: dict (asset-class-specific details)
        """
        ...

    @abstractmethod
    def evaluate_exit(self, position: Dict) -> Optional[Dict]:
        """Check if an open position should be exited.

        Returns:
            Exit signal dict if exit warranted, None otherwise.
        """
        ...

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return the list of feature names this strategy provides for ML."""
        ...
```

### 4.3 Generalized Data Provider Protocol

```python
# shared/provider_protocol.py (extended)

from typing import Dict, List, Protocol, runtime_checkable
import pandas as pd
from shared.asset_class import AssetClass


@runtime_checkable
class DataProvider(Protocol):
    """Contract that all market-data providers must satisfy."""

    asset_class: AssetClass

    def get_quote(self, instrument: str) -> Dict:
        """Get a real-time quote."""
        ...

    def get_historical(
        self, instrument: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        ...


@runtime_checkable
class OptionsDataProvider(DataProvider, Protocol):
    """Extended protocol for options-specific data."""

    def get_options_chain(self, ticker: str, expiration: str) -> pd.DataFrame: ...
    def get_expirations(self, ticker: str) -> List[str]: ...
    def get_full_chain(self, ticker: str, min_dte: int, max_dte: int) -> pd.DataFrame: ...


@runtime_checkable
class FuturesDataProvider(DataProvider, Protocol):
    """Extended protocol for futures-specific data."""

    def get_term_structure(self, root_symbol: str) -> pd.DataFrame: ...
    def get_contract_specs(self, symbol: str) -> Dict: ...
    def get_volume_profile(self, symbol: str, period: str) -> pd.DataFrame: ...


@runtime_checkable
class CryptoDataProvider(DataProvider, Protocol):
    """Extended protocol for crypto-specific data."""

    def get_funding_rate(self, symbol: str) -> Dict: ...
    def get_order_book(self, symbol: str, depth: int) -> Dict: ...
    def get_on_chain_metrics(self, symbol: str) -> Dict: ...


@runtime_checkable
class PredictionMarketDataProvider(DataProvider, Protocol):
    """Extended protocol for prediction market data."""

    def get_markets(self, active: bool, limit: int) -> List[Dict]: ...
    def get_market_detail(self, market_id: str) -> Dict: ...
    def get_order_book(self, token_id: str) -> Dict: ...
```

### 4.4 Multi-Asset Orchestrator

```python
# orchestrator.py

class MultiAssetOrchestrator:
    """Coordinates scanning across all asset classes."""

    def __init__(self, config: Dict):
        self.config = config
        self.strategies: Dict[AssetClass, BaseStrategy] = {}
        self.ml_pipelines: Dict[AssetClass, MLPipeline] = {}
        self.executors: Dict[AssetClass, BaseExecutor] = {}
        self.portfolio_manager = PortfolioManager(config)
        self.unified_scorer = UnifiedScorer()

    def register_asset_class(
        self,
        asset_class: AssetClass,
        strategy: BaseStrategy,
        ml_pipeline: MLPipeline,
        executor: BaseExecutor,
    ):
        """Register a new asset class module."""
        self.strategies[asset_class] = strategy
        self.ml_pipelines[asset_class] = ml_pipeline
        self.executors[asset_class] = executor

    async def scan_all(self) -> List[Dict]:
        """Run scans across all registered asset classes concurrently."""
        # 1. Parallel scan each asset class
        # 2. ML-enhance each opportunity
        # 3. Normalize scores via UnifiedScorer
        # 4. Portfolio risk check
        # 5. Auto-execute qualifying signals (score >= 60)
        ...

    async def scan_asset_class(self, asset_class: AssetClass) -> List[Dict]:
        """Scan a single asset class."""
        ...
```

### 4.5 Config Extension

```yaml
# config.yaml additions

# Asset Classes
asset_classes:
  options:
    enabled: true
    # ... existing options config stays as-is ...

  futures:
    enabled: false  # Enable when ready
    instruments:
      - symbol: ES
        name: "E-mini S&P 500"
        exchange: CME
        tick_size: 0.25
        tick_value: 12.50
        margin_initial: 12650
        margin_maintenance: 11500
      - symbol: NQ
        name: "E-mini Nasdaq 100"
        exchange: CME
        tick_size: 0.25
        tick_value: 5.00
        margin_initial: 17600
        margin_maintenance: 16000
      - symbol: CL
        name: "Crude Oil"
        exchange: NYMEX
        tick_size: 0.01
        tick_value: 10.00
        margin_initial: 7800
        margin_maintenance: 6500
    strategy:
      max_leverage: 5.0
      strategies: ["mean_reversion", "momentum", "calendar_spread"]
      lookback_bars: 200
      entry_z_score: 2.0        # Mean reversion entry threshold
      momentum_period: 20        # Momentum lookback
    risk:
      max_contracts: 5
      stop_loss_ticks: 20       # ES: 20 ticks = $250/contract
      profit_target_ticks: 40
      max_daily_loss: 2000
    data:
      provider: "databento"     # or "polygon_futures"
      api_key: "${DATABENTO_API_KEY}"

  crypto:
    enabled: false
    instruments:
      - symbol: BTC
        spot_pair: "BTC/USDT"
        perp_pair: "BTC/USDT:USDT"
      - symbol: ETH
        spot_pair: "ETH/USDT"
        perp_pair: "ETH/USDT:USDT"
      - symbol: SOL
        spot_pair: "SOL/USDT"
        perp_pair: "SOL/USDT:USDT"
    strategy:
      max_leverage: 3.0
      strategies: ["funding_arb", "momentum", "mean_reversion", "breakout"]
      funding_rate_threshold: 0.03  # 3% annualized
      momentum_period: 14
    risk:
      max_position_usd: 10000
      stop_loss_pct: 3.0
      take_profit_pct: 6.0
      max_daily_loss: 3000
    data:
      exchange: "binance"       # Primary exchange
      fallback_exchange: "coinbase"
      api_key: "${BINANCE_API_KEY}"
      api_secret: "${BINANCE_API_SECRET}"

  prediction_markets:
    enabled: false
    strategy:
      min_edge: 0.05            # 5% minimum edge to trade
      min_liquidity: 10000      # $10k minimum market liquidity
      max_position_per_market: 500
      categories: ["politics", "crypto", "sports", "finance"]
    risk:
      max_total_exposure: 5000
      max_per_market: 500
      min_days_to_resolution: 3
    data:
      api_url: "https://clob.polymarket.com"
      chain_id: 137             # Polygon network

# Portfolio-Level Risk (NEW)
portfolio:
  max_total_leverage: 3.0       # Across all assets
  max_correlation_overlap: 0.70 # Don't stack correlated positions
  max_drawdown_pct: 15.0        # Kill switch threshold
  rebalance_interval: "daily"
  allocation_limits:
    options: 0.40               # Max 40% in options
    futures: 0.30               # Max 30% in futures
    crypto: 0.20                # Max 20% in crypto
    prediction_markets: 0.10    # Max 10% in predictions
```

---

## 5. Phase 2: Futures Module

### 5.1 Why Futures First

- Most structurally similar to options (same underlyings — SPY→ES, QQQ→NQ)
- Existing regime detector and technical analysis directly applicable
- Well-understood market microstructure
- CME data is institutional-grade and reliable
- 23h/day trading extends opportunity set beyond market hours

### 5.2 Futures Data Provider

```python
# strategy/providers/futures_provider.py

class FuturesDataProvider:
    """Futures market data from Databento or Polygon."""

    def __init__(self, config: Dict):
        self.api_key = config.get("api_key")
        self.provider = config.get("provider", "databento")

    def get_quote(self, symbol: str) -> Dict:
        """Real-time futures quote with bid/ask/last/volume."""
        ...

    def get_historical(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """OHLCV bars. Intervals: 1m, 5m, 15m, 1h, 1d."""
        ...

    def get_term_structure(self, root_symbol: str) -> pd.DataFrame:
        """All active contracts with expiration, price, volume, open interest.

        Returns DataFrame:
            contract | expiration | last | bid | ask | volume | oi | days_to_expiry
            ESH26    | 2026-03-20 | 5120 | 5119.75 | 5120.25 | 1.2M | 2.8M | 30
            ESM26    | 2026-06-19 | 5135 | 5134.50 | 5135.50 | 400K | 1.1M | 121
        """
        ...

    def get_volume_profile(self, symbol: str, period: str) -> pd.DataFrame:
        """Price levels with volume concentration (POC, VAH, VAL)."""
        ...

    def get_contract_specs(self, symbol: str) -> Dict:
        """Tick size, tick value, margin requirements, trading hours."""
        ...
```

### 5.3 Futures Strategy Module

Three sub-strategies, each scored independently then ensembled:

#### A. Mean Reversion Strategy
```
Signal: When price deviates >2 standard deviations from VWAP or 20-period mean
Entry: Z-score > 2.0 (short) or < -2.0 (long) on 15-min bars
Exit: Z-score returns to 0 or stop at 3.0 standard deviations
Best in: Low-vol trending or mean-reverting regimes
Typical hold: 1-4 hours
```

#### B. Momentum/Trend Strategy
```
Signal: Breakout above/below consolidation range with volume confirmation
Entry: Price breaks N-bar high/low with volume > 1.5x average
Exit: Trailing stop at 2x ATR
Best in: High-vol trending regimes
Typical hold: 1-5 days
```

#### C. Calendar Spread Strategy
```
Signal: Term structure anomalies (contango/backwardation extremes)
Entry: When front-back spread deviates >2σ from historical mean
Exit: Spread mean-reverts or time stop at 10 days
Best in: Any regime (market neutral)
Typical hold: 3-10 days
```

### 5.4 Futures Feature Engineering

```python
# Features specific to futures (extends base FeatureEngine)

class FuturesFeatureEngine(FeatureEngine):
    """Features for futures ML models."""

    FUTURES_FEATURES = [
        # Term structure
        "front_back_spread",          # Price difference front vs back month
        "term_structure_slope",       # Contango (+) vs backwardation (-)
        "roll_yield_annualized",      # Implied carry return
        "basis_vs_spot",              # Futures premium/discount to spot

        # Volume/flow
        "volume_vs_20d_avg",          # Relative volume
        "oi_change_1d",               # Open interest change (position building)
        "oi_change_5d",
        "large_trader_net",           # COT positioning (weekly)
        "commercial_net",             # Commercial hedger positioning

        # Microstructure
        "bid_ask_spread_ticks",       # Liquidity measure
        "volume_profile_poc",         # Point of control (highest volume price)
        "dist_from_poc_pct",          # Distance from POC
        "vah_val_range",              # Value area width

        # Cross-market
        "es_nq_ratio",               # Equity index relative strength
        "es_cl_correlation_20d",     # Cross-asset correlation
        "dollar_index_return_5d",    # DXY impact on commodities
        "yield_curve_slope",         # 10Y-2Y treasury spread

        # Intraday patterns
        "time_of_day_bucket",        # 4 sessions: Asia/Europe/US/overnight
        "day_of_week",
        "is_rollover_week",          # Heightened vol near contract roll
        "days_to_expiration",        # Current front month DTE
    ]
```

### 5.5 Futures ML Model

```python
# ml/futures_model.py

class FuturesMLModel:
    """ML model for futures trade prediction.

    Architecture: Gradient Boosted Trees (XGBoost) + LSTM ensemble

    XGBoost: Handles tabular features (term structure, COT, technicals)
    LSTM: Captures sequential patterns in price/volume series

    Target: 3-class classification
        - 0 = losing trade (< -1 ATR move against)
        - 1 = scratch/small (within ±1 ATR)
        - 2 = winning trade (> 1 ATR move in direction)
    """

    def __init__(self, config: Dict):
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
        )
        # LSTM for sequence features (price bars)
        self.use_lstm = config.get("use_lstm", False)
        self.sequence_length = 50  # 50 bars lookback

    def predict(self, features: Dict, price_sequence: Optional[np.ndarray] = None) -> Dict:
        """Predict trade outcome.

        Returns:
            {
                "prediction": int,          # 0, 1, or 2
                "probabilities": [float],   # [p_loss, p_scratch, p_win]
                "win_probability": float,   # p_win
                "confidence": float,        # max(probabilities)
                "signal": str,              # "strong_long" | "long" | "neutral" | "short" | "strong_short"
                "expected_value": float,    # probability-weighted expected PnL
            }
        """
        ...

    def train(self, features_df: pd.DataFrame, labels: pd.Series,
              price_sequences: Optional[np.ndarray] = None) -> Dict:
        """Train on historical futures data."""
        ...
```

### 5.6 Futures Execution

```python
# execution/futures_executor.py

class FuturesExecutor(BaseExecutor):
    """Futures order execution via broker API.

    Supported brokers (in priority order):
    1. Alpaca (if futures support available) — already integrated
    2. Interactive Brokers via ib_insync
    3. Tradovate API
    """

    def place_order(self, order: Dict) -> Dict:
        """Place a futures order.

        Args:
            order: {
                "symbol": "ESH26",
                "side": "buy" | "sell",
                "quantity": 1,
                "order_type": "limit" | "market" | "stop",
                "limit_price": 5120.50,     # for limit orders
                "stop_price": 5100.00,      # for stop orders
                "time_in_force": "day" | "gtc",
            }
        """
        ...

    def place_bracket_order(self, entry: Dict, take_profit: Dict, stop_loss: Dict) -> Dict:
        """Place entry + TP + SL as OCO bracket."""
        ...

    def get_positions(self) -> List[Dict]:
        """Current open futures positions with unrealized PnL."""
        ...

    def get_account(self) -> Dict:
        """Account balance, margin used, margin available."""
        ...
```

---

## 6. Phase 3: Crypto Module

### 6.1 Why Crypto Second

- 24/7 market = continuous opportunity (complements options/futures hours)
- Funding rate arbitrage is a quantifiable, repeatable edge
- On-chain data provides unique alpha signals not available in TradFi
- Multiple exchange APIs with good libraries (ccxt)
- Higher volatility = more ML signal to capture

### 6.2 Crypto Data Provider

```python
# strategy/providers/crypto_provider.py

import ccxt  # Unified exchange library

class CryptoDataProvider:
    """Crypto market data from exchanges via ccxt."""

    def __init__(self, config: Dict):
        exchange_id = config.get("exchange", "binance")
        self.exchange = getattr(ccxt, exchange_id)({
            "apiKey": config.get("api_key"),
            "secret": config.get("api_secret"),
            "sandbox": config.get("sandbox", True),
        })

    def get_quote(self, symbol: str) -> Dict:
        """Spot + perpetual prices with funding rate."""
        ...

    def get_historical(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """OHLCV candles. Timeframes: 1m, 5m, 15m, 1h, 4h, 1d."""
        ...

    def get_funding_rate(self, symbol: str) -> Dict:
        """Current and historical funding rates.

        Returns:
            {
                "current_rate": 0.0001,              # 0.01% per 8h
                "annualized_rate": 0.0438,            # 4.38%/year
                "next_funding_time": "2026-02-18T16:00:00Z",
                "predicted_rate": 0.00012,
                "historical_avg_30d": 0.00008,
                "historical_avg_90d": 0.00006,
            }
        """
        ...

    def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """L2 order book with bid/ask depth.

        Returns:
            {
                "bids": [[price, size], ...],
                "asks": [[price, size], ...],
                "spread_bps": 1.5,
                "bid_depth_usd": 2500000,
                "ask_depth_usd": 2100000,
                "imbalance": 0.087,   # positive = more bids
            }
        """
        ...

    def get_on_chain_metrics(self, symbol: str) -> Dict:
        """On-chain data from Glassnode/CryptoQuant APIs.

        Returns:
            {
                "exchange_netflow_24h": -1500.5,      # BTC leaving exchanges (bullish)
                "exchange_reserve": 2100000,
                "whale_transactions_24h": 342,
                "active_addresses_24h": 850000,
                "nvt_ratio": 65.2,                    # Network Value to Transactions
                "mvrv_ratio": 1.8,                    # Market Value to Realized Value
                "puell_multiple": 0.9,
                "sopr": 1.02,                         # Spent Output Profit Ratio
                "long_short_ratio": 1.35,             # Derivatives positioning
                "open_interest_usd": 15000000000,
                "oi_change_24h_pct": 2.5,
            }
        """
        ...
```

### 6.3 Crypto Strategy Module

Four sub-strategies:

#### A. Funding Rate Arbitrage
```
Signal: Funding rate significantly above/below equilibrium
Entry: When annualized funding > 15% (long spot + short perp)
       When annualized funding < -10% (short spot + long perp)
Exit: Funding normalizes to < 5% or after funding payment collected
Edge: Captures 10-40% annualized risk-free-ish yield
Risk: Basis risk if spot and perp diverge; exchange counterparty risk
```

#### B. Momentum/Trend Following
```
Signal: Price breakout with volume + on-chain confirmation
Entry: Price above 20-day high with exchange outflows (accumulation)
Exit: Trailing stop at 3x ATR or regime shift to crisis
Best in: Bull markets, high-vol trending
Typical hold: 3-14 days
```

#### C. Mean Reversion
```
Signal: Oversold/overbought extremes with order book support
Entry: RSI < 25 with positive order book imbalance
Exit: RSI returns to 50 or stop at new low
Best in: Range-bound markets
Typical hold: 1-3 days
```

#### D. Breakout/Breakdown
```
Signal: Price consolidation squeeze (Bollinger %B < 0.1 or > 0.9)
Entry: Break of consolidation range with volume > 2x average
Exit: Measured move (range width) or trailing stop
Best in: After low-volatility regimes
Typical hold: 1-7 days
```

### 6.4 Crypto Feature Engineering

```python
class CryptoFeatureEngine(FeatureEngine):
    """Features specific to cryptocurrency markets."""

    CRYPTO_FEATURES = [
        # Funding & derivatives
        "funding_rate_current",
        "funding_rate_annualized",
        "funding_rate_vs_30d_avg",
        "open_interest_usd",
        "oi_change_24h_pct",
        "long_short_ratio",
        "liquidation_volume_24h",

        # On-chain
        "exchange_netflow_24h",        # Negative = accumulation (bullish)
        "exchange_reserve_change_7d",
        "whale_tx_count_24h",
        "active_addresses_vs_30d",
        "nvt_ratio",                   # High = overvalued
        "mvrv_ratio",                  # >3 = overheated, <1 = undervalued
        "sopr",                        # >1 = profit taking, <1 = capitulation
        "puell_multiple",              # Miner revenue vs 365d avg

        # Order book / microstructure
        "bid_ask_spread_bps",
        "order_book_imbalance",
        "bid_depth_usd",
        "ask_depth_usd",

        # Cross-crypto
        "btc_dominance",
        "btc_eth_correlation_30d",
        "alt_season_index",            # Whether alts outperforming BTC

        # Macro / sentiment
        "fear_greed_index",            # Crypto Fear & Greed (0-100)
        "google_trends_btc",           # Retail interest proxy
        "tether_market_cap_change_7d", # Stablecoin flow indicator

        # Technical (inherited from base)
        # RSI, MACD, Bollinger, ATR, momentum, etc.
    ]
```

### 6.5 Crypto ML Model

```python
# ml/crypto_model.py

class CryptoMLModel:
    """ML model for crypto trade prediction.

    Architecture: XGBoost + LightGBM ensemble with on-chain feature importance

    Key insight: On-chain features (exchange flows, MVRV, SOPR) have
    predictive power at 1-7 day horizons that doesn't exist in TradFi.

    Target: Regression predicting N-day forward return
        - Binned into: strong_short, short, neutral, long, strong_long
    """

    def __init__(self, config: Dict):
        self.xgb_model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
        )
        self.lgb_model = None  # LightGBM for ensemble diversity
        self.prediction_horizon = config.get("horizon_days", 3)

    def predict(self, features: Dict) -> Dict:
        """Predict forward return and classify signal.

        Returns:
            {
                "predicted_return": 0.035,    # 3.5% expected return
                "return_std": 0.042,          # Uncertainty
                "signal": "long",
                "signal_strength": 0.72,
                "confidence": 0.68,
                "feature_importance": {       # Top 5 driving features
                    "exchange_netflow_24h": 0.15,
                    "funding_rate_vs_30d_avg": 0.12,
                    ...
                }
            }
        """
        ...
```

### 6.6 Crypto Execution

```python
# execution/crypto_executor.py

class CryptoExecutor(BaseExecutor):
    """Crypto order execution via exchange APIs (ccxt).

    Supports:
    - Spot market/limit orders
    - Perpetual futures with leverage
    - Funding rate arbitrage (simultaneous spot + perp)
    """

    def place_spot_order(self, order: Dict) -> Dict: ...
    def place_perp_order(self, order: Dict) -> Dict: ...

    def place_funding_arb(self, symbol: str, size_usd: float, direction: str) -> Dict:
        """Place funding rate arbitrage: long spot + short perp (or inverse).

        This is the lowest-risk crypto strategy — delta-neutral yield capture.
        """
        ...

    def get_positions(self) -> List[Dict]:
        """All open spot + perp positions with unrealized PnL."""
        ...
```

---

## 7. Phase 4: Prediction Markets Module

### 7.1 Why Prediction Markets Last

- Highest potential edge (inefficient markets with retail participants)
- Most different from existing system (event-based, not price-based)
- Requires different ML approach (probability estimation, not price prediction)
- Polymarket API is newer and less documented
- Smallest capital allocation (10% of portfolio)

### 7.2 Polymarket Data Provider

```python
# strategy/providers/polymarket_provider.py

class PolymarketProvider:
    """Polymarket CLOB (Central Limit Order Book) API integration.

    Polymarket uses a CLOB on Polygon (Ethereum L2) with
    conditional tokens (YES/NO shares priced $0.00-$1.00).

    Key endpoints:
    - GET /markets — list active markets
    - GET /markets/{id} — market detail with current prices
    - GET /book — order book for a token
    - POST /order — place an order
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, config: Dict):
        self.api_key = config.get("api_key")
        self.chain_id = config.get("chain_id", 137)

    def get_markets(
        self,
        active: bool = True,
        category: Optional[str] = None,
        min_liquidity: float = 10000,
        limit: int = 100,
    ) -> List[Dict]:
        """Fetch active markets with filtering.

        Returns list of:
            {
                "id": "0x...",
                "question": "Will BTC exceed $100K by March 2026?",
                "category": "crypto",
                "end_date": "2026-03-31",
                "liquidity_usd": 250000,
                "volume_24h": 15000,
                "yes_price": 0.72,          # Market thinks 72% likely
                "no_price": 0.28,
                "tokens": [
                    {"token_id": "0x...yes", "outcome": "Yes", "price": 0.72},
                    {"token_id": "0x...no", "outcome": "No", "price": 0.28},
                ],
                "resolution_source": "Official BTC price on CoinGecko",
            }
        """
        ...

    def get_market_history(self, market_id: str) -> pd.DataFrame:
        """Price history for a market (for trend detection).

        Returns DataFrame:
            timestamp | yes_price | no_price | volume
        """
        ...

    def get_order_book(self, token_id: str) -> Dict:
        """L2 order book for a specific token (YES or NO)."""
        ...

    def get_related_markets(self, market_id: str) -> List[Dict]:
        """Find correlated/related markets for cross-market analysis."""
        ...
```

### 7.3 Prediction Market Strategy

The core edge: Polymarket prices often lag public information, especially in:
- News-driven events (election polls, regulatory decisions)
- Cross-market implications (if market A resolves YES, market B becomes more likely)
- Anchoring bias (prices sticky near round numbers like 0.50)

```python
# strategy/prediction_strategy.py

class PredictionMarketStrategy(BaseStrategy):
    """Identify mispriced prediction markets.

    Edge sources:
    1. Model-vs-market divergence: Our model estimates P(event) differently
    2. Cross-market arbitrage: Related markets with inconsistent prices
    3. Information lag: Market hasn't incorporated recent news/data
    4. Liquidity premium: Illiquid markets offer better prices
    """

    def scan(self, instruments: List[str], config: Dict) -> List[Dict]:
        """Scan Polymarket for mispriced markets.

        Process:
        1. Fetch all active markets meeting liquidity threshold
        2. For each market, estimate "fair" probability using ML model
        3. Compare model probability to market price
        4. If |model_prob - market_price| > min_edge, flag as opportunity
        5. Check for cross-market arbitrage
        6. Score by edge size, liquidity, and time to resolution
        """
        ...

    def estimate_fair_probability(self, market: Dict) -> Dict:
        """Estimate the fair probability of a market outcome.

        Uses:
        - Category-specific models (politics, crypto, sports, finance)
        - Historical base rates for similar events
        - Current data (polls, prices, statistics)
        - Cross-market implied probabilities

        Returns:
            {
                "model_probability": 0.65,
                "market_price": 0.72,
                "edge": -0.07,           # Negative = market overprices YES
                "confidence": 0.75,
                "reasoning": ["...", "..."],
                "data_sources": ["polls", "historical_base_rate"],
            }
        """
        ...
```

### 7.4 Prediction Market Feature Engineering

```python
class PredictionFeatureEngine(FeatureEngine):
    """Features for prediction market ML models."""

    PREDICTION_FEATURES = [
        # Market structure
        "current_yes_price",
        "current_no_price",
        "bid_ask_spread",
        "liquidity_usd",
        "volume_24h",
        "volume_7d",

        # Price dynamics
        "price_momentum_1d",          # YES price change
        "price_momentum_7d",
        "price_volatility_7d",
        "max_price_30d",              # Recent high
        "min_price_30d",              # Recent low

        # Time
        "days_to_resolution",
        "pct_time_elapsed",           # How far through the event timeline

        # Category-specific
        "category_encoded",           # One-hot: politics, crypto, sports, etc.
        "historical_base_rate",       # How often similar events resolve YES

        # Cross-market
        "related_market_implied_prob", # Probability implied by related markets
        "cross_market_consistency",    # Do related markets agree?

        # Market sentiment
        "comment_count",              # Social engagement proxy
        "unique_traders",             # Participation breadth
        "large_order_flow",           # Whale activity
    ]
```

### 7.5 Prediction Market ML Model

```python
# ml/prediction_model.py

class PredictionMLModel:
    """ML model for prediction market probability estimation.

    Architecture: Category-specific ensemble

    For each category (politics, crypto, sports, finance):
    - XGBoost probability estimator
    - Calibrated using Platt scaling
    - Historical base rate prior

    Key challenge: Each market is unique — transfer learning between
    similar market types is more important than fitting to specific markets.
    """

    def __init__(self, config: Dict):
        self.category_models: Dict[str, XGBClassifier] = {}
        self.calibrators: Dict[str, CalibratedClassifierCV] = {}
        self.base_rates: Dict[str, float] = {
            "politics": 0.50,
            "crypto": 0.45,
            "sports": 0.50,
            "finance": 0.48,
        }

    def estimate_probability(self, market: Dict, features: Dict) -> Dict:
        """Estimate fair probability for a market.

        Returns:
            {
                "model_probability": 0.65,
                "calibrated_probability": 0.63,
                "confidence": 0.72,
                "edge_vs_market": 0.63 - market["yes_price"],
                "kelly_bet_size": 0.08,
            }
        """
        ...
```

---

## 8. ML Model Architectures — Detailed Design

### 8.1 Model Selection Rationale

| Asset Class | Primary Model | Secondary Model | Why |
|---|---|---|---|
| Options | XGBoost Classifier | — | Tabular features, binary outcome, existing system |
| Futures | XGBoost Classifier | LSTM (optional) | Tabular + sequential patterns, 3-class |
| Crypto | XGBoost Regressor | LightGBM | Regression target, on-chain features important |
| Prediction | XGBoost Classifier | Calibrated ensemble | Probability estimation, category-specific |

### 8.2 Ensemble Architecture

```python
# ml/ensemble.py

class MultiModelEnsemble:
    """Combines predictions from multiple models for a single asset class.

    Ensemble methods:
    1. Simple average (default)
    2. Weighted average (based on historical accuracy)
    3. Stacking (meta-learner on top of base model predictions)
    """

    def __init__(self, models: List, method: str = "weighted_average"):
        self.models = models
        self.method = method
        self.weights = [1.0 / len(models)] * len(models)  # Equal initially

    def predict(self, features: Dict) -> Dict:
        """Get ensemble prediction."""
        predictions = [m.predict(features) for m in self.models]

        if self.method == "weighted_average":
            combined_prob = sum(
                w * p["probability"] for w, p in zip(self.weights, predictions)
            )
            return {
                "probability": combined_prob,
                "confidence": min(p["confidence"] for p in predictions),
                "model_agreement": self._calculate_agreement(predictions),
                "individual_predictions": predictions,
            }
        ...

    def update_weights(self, actual_outcomes: pd.Series):
        """Update ensemble weights based on recent performance."""
        ...
```

### 8.3 Training Data Strategy

| Asset Class | Training Data Source | Volume | Update Frequency |
|---|---|---|---|
| Options | Historical trades (existing) + synthetic | 2000+ samples | Weekly retrain |
| Futures | Databento historical bars + COT reports | 5+ years daily | Weekly retrain |
| Crypto | Exchange historical + Glassnode on-chain | 3+ years daily | Daily retrain (24/7 market) |
| Prediction | Polymarket historical resolutions | 1000+ resolved markets | Per-batch |

### 8.4 Feature Importance Monitoring

```python
class FeatureMonitor:
    """Track feature importance drift and data quality.

    Alerts when:
    - Feature importance shifts significantly (>20% relative change)
    - Feature values drift out of training distribution
    - Model performance degrades (rolling accuracy < threshold)
    """

    def check_drift(self, current_features: Dict, model: str) -> Dict:
        """Compare current feature distributions to training baseline."""
        ...

    def get_importance_report(self, model: str) -> pd.DataFrame:
        """Feature importance rankings over time."""
        ...
```

---

## 9. Unified Scoring System

### 9.1 Problem

Each asset class has different risk/reward profiles, time horizons, and signal characteristics. A "75 score" in options (87% win rate, 30% return) is very different from a "75 score" in crypto (60% win rate, 200% return). We need a unified scale.

### 9.2 Solution: Risk-Adjusted Expected Value Scoring

```python
# ml/unified_scorer.py

class UnifiedScorer:
    """Normalize scores across asset classes into a unified 0-100 scale.

    The unified score represents: "How good is this opportunity relative to
    all opportunities across all asset classes?"

    Formula:
        unified_score = w1 * ev_score + w2 * confidence_score + w3 * risk_score

    Where:
        ev_score = percentile_rank(expected_value / max_drawdown)
        confidence_score = model_confidence * model_accuracy_30d
        risk_score = (1 - correlation_to_portfolio) * (1 - leverage_used/max_leverage)
    """

    def __init__(self):
        self.historical_scores: Dict[AssetClass, List[float]] = {
            ac: [] for ac in AssetClass
        }

    def score(
        self,
        raw_score: float,
        asset_class: AssetClass,
        ml_probability: float,
        ml_confidence: float,
        expected_return: float,
        max_loss: float,
        leverage: float,
        correlation_to_portfolio: float,
    ) -> float:
        """Calculate unified cross-asset score (0-100).

        Args:
            raw_score: Strategy-specific score (0-100)
            asset_class: Which asset class
            ml_probability: ML win/success probability
            ml_confidence: Model confidence in prediction
            expected_return: Expected return if trade works
            max_loss: Maximum loss scenario
            leverage: Leverage being used
            correlation_to_portfolio: Correlation to existing portfolio

        Returns:
            Unified score (0-100) comparable across asset classes
        """
        # 1. Expected value component (40% weight)
        ev = ml_probability * expected_return + (1 - ml_probability) * (-abs(max_loss))
        ev_ratio = ev / abs(max_loss) if max_loss != 0 else 0
        ev_score = self._sigmoid_scale(ev_ratio, center=0.1, steepness=10) * 100

        # 2. Confidence component (30% weight)
        confidence_score = ml_confidence * 100

        # 3. Risk component (30% weight)
        leverage_penalty = leverage / ASSET_CLASS_CONFIGS[asset_class].max_leverage
        correlation_penalty = correlation_to_portfolio
        risk_score = (1 - 0.5 * leverage_penalty - 0.5 * correlation_penalty) * 100

        unified = 0.40 * ev_score + 0.30 * confidence_score + 0.30 * risk_score
        return round(max(0, min(100, unified)), 1)

    def _sigmoid_scale(self, x: float, center: float, steepness: float) -> float:
        """Sigmoid scaling for smooth 0-1 mapping."""
        return 1 / (1 + np.exp(-steepness * (x - center)))
```

### 9.3 Execution Thresholds

| Unified Score | Action | Asset Classes |
|---|---|---|
| 80-100 | **Auto-execute** (strong signal) | All |
| 60-79 | **Auto-execute** (standard signal) | All |
| 40-59 | **Alert only** (human review) | All |
| 0-39 | **Skip** (below threshold) | All |

---

## 10. Portfolio-Level Risk Management

### 10.1 Portfolio Manager

```python
# risk/portfolio_manager.py

class PortfolioManager:
    """Cross-asset portfolio risk management.

    Responsibilities:
    1. Track total portfolio exposure across all asset classes
    2. Enforce allocation limits per asset class
    3. Monitor cross-asset correlations
    4. Manage aggregate leverage
    5. Drawdown-based kill switch
    6. Position-level risk budgeting
    """

    def __init__(self, config: Dict):
        self.account_size = config["risk"]["account_size"]
        self.max_drawdown = config["portfolio"]["max_drawdown_pct"] / 100
        self.allocation_limits = config["portfolio"]["allocation_limits"]
        self.max_total_leverage = config["portfolio"]["max_total_leverage"]

        self.correlation_tracker = CorrelationTracker()
        self.leverage_controller = LeverageController(config)
        self.drawdown_monitor = DrawdownMonitor(self.max_drawdown)

    def can_open_position(self, opportunity: Dict) -> Tuple[bool, str]:
        """Check if a new position is allowed given portfolio constraints.

        Checks (in order):
        1. Kill switch not engaged
        2. Drawdown within limits
        3. Asset class allocation not exceeded
        4. Total leverage within limits
        5. Correlation check (new position not too correlated with existing)
        6. Individual position size within limits
        """
        ...

    def calculate_position_size(
        self,
        opportunity: Dict,
        ml_result: Dict,
    ) -> float:
        """Calculate risk-adjusted position size considering portfolio context.

        Base size from Kelly criterion, then adjusted for:
        - Current asset class utilization (reduce as approaching limit)
        - Portfolio leverage headroom
        - Correlation with existing positions
        - Current drawdown level (reduce size during drawdowns)
        """
        ...

    def get_portfolio_snapshot(self) -> Dict:
        """Current portfolio state across all asset classes.

        Returns:
            {
                "total_value": 105000,
                "total_pnl": 5000,
                "total_pnl_pct": 5.0,
                "total_leverage": 2.1,
                "current_drawdown_pct": 1.2,
                "positions_by_asset_class": {
                    "options": {"count": 3, "value": 42000, "pnl": 2100, "allocation_pct": 40.0},
                    "futures": {"count": 1, "value": 25000, "pnl": 1500, "allocation_pct": 23.8},
                    "crypto": {"count": 2, "value": 18000, "pnl": 900, "allocation_pct": 17.1},
                    "prediction_markets": {"count": 4, "value": 3000, "pnl": 500, "allocation_pct": 2.9},
                },
                "correlation_matrix": {...},
                "risk_metrics": {
                    "var_95": -3200,
                    "expected_shortfall": -4800,
                    "sharpe_ratio_30d": 1.8,
                },
            }
        """
        ...
```

### 10.2 Correlation Tracker

```python
# risk/correlation_tracker.py

class CorrelationTracker:
    """Track rolling correlations across all positions and asset classes.

    Uses 30-day rolling returns to compute:
    - Pairwise instrument correlations
    - Asset-class-level correlations
    - Portfolio concentration score
    """

    def __init__(self, lookback_days: int = 30):
        self.lookback = lookback_days
        self.returns_cache: Dict[str, pd.Series] = {}

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Full correlation matrix across all active instruments."""
        ...

    def check_new_position_correlation(
        self, instrument: str, existing_positions: List[Dict]
    ) -> Dict:
        """Check if adding a new position would increase portfolio concentration.

        Returns:
            {
                "max_correlation": 0.85,
                "avg_correlation": 0.45,
                "most_correlated_with": "ES",
                "diversification_benefit": 0.15,
                "recommendation": "reduce_size",  # or "proceed" or "skip"
            }
        """
        ...
```

### 10.3 Leverage Controller

```python
# risk/leverage_controller.py

class LeverageController:
    """Manage leverage across the portfolio.

    Rules:
    - Each asset class has its own max leverage
    - Total portfolio leverage cannot exceed configured max (default 3x)
    - Leverage reduces automatically during drawdowns
    - No leverage allowed when drawdown > 10%
    """

    def get_current_leverage(self) -> Dict:
        """Current leverage by asset class and total.

        Returns:
            {
                "options": 1.0,      # Options are defined risk, always 1x
                "futures": 4.2,      # 4.2x on futures positions
                "crypto": 2.8,       # 2.8x on crypto perps
                "prediction_markets": 1.0,
                "portfolio_total": 2.3,
                "headroom": 0.7,     # Can add 0.7x more
            }
        """
        ...

    def max_allowed_leverage(self, asset_class: AssetClass) -> float:
        """Max leverage allowed for new position given current state.

        Scales down linearly:
        - At 0% drawdown: full leverage allowed
        - At 5% drawdown: 75% of max
        - At 10% drawdown: 50% of max
        - At 15%+ drawdown: no leverage (1x only, or kill switch)
        """
        ...
```

### 10.4 Drawdown Monitor

```python
# risk/drawdown_monitor.py

class DrawdownMonitor:
    """Portfolio-wide drawdown monitoring and circuit breaker.

    Levels:
    - Warning (5%): Log warning, reduce new position sizes by 25%
    - Caution (10%): Alert via Telegram, reduce sizes by 50%, no new leveraged trades
    - Critical (15%): KILL SWITCH — close all positions, halt all trading
    """

    def __init__(self, max_drawdown: float):
        self.max_drawdown = max_drawdown
        self.peak_value = 0
        self.current_value = 0

    def update(self, portfolio_value: float) -> Dict:
        """Update with latest portfolio value and check thresholds.

        Returns:
            {
                "current_drawdown_pct": 3.2,
                "peak_value": 108000,
                "current_value": 104544,
                "level": "normal",    # normal | warning | caution | critical
                "size_multiplier": 1.0,
                "actions": [],        # List of automated actions taken
            }
        """
        ...
```

---

## 11. Data Pipeline Design

### 11.1 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ Polygon  │Databento │ Binance  │Polymarket│  Glassnode          │
│ (Options)│(Futures) │ (Crypto) │ (PredMkt)│  (On-chain)         │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────┬──────────────┘
     │          │          │          │            │
     ▼          ▼          ▼          ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                         │
│                                                                  │
│  Circuit breaker │ Rate limiting │ Retry logic │ Schema validate │
│                                                                  │
│  Each provider implements the DataProvider protocol              │
│  Parallel ingestion via ThreadPoolExecutor                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA CACHE LAYER                           │
│                                                                  │
│  shared/data_cache.py (extended)                                │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ In-memory   │  │ SQLite       │  │ Disk cache             │ │
│  │ (hot data)  │  │ (warm data)  │  │ (historical archives)  │ │
│  │ TTL: 1-15m  │  │ TTL: 1-24h  │  │ TTL: permanent         │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE COMPUTATION                            │
│                                                                  │
│  Base features (shared)  +  Asset-class-specific features        │
│  Computed on-demand per scan, cached for duration of scan        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML MODEL INFERENCE                             │
│                                                                  │
│  Per-asset model prediction → Ensemble → Unified scoring         │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Scan Scheduling

```python
# shared/scheduler.py (extended)

SCAN_SCHEDULE = {
    AssetClass.OPTIONS: {
        "active_hours": "09:30-16:00 ET",
        "interval_minutes": 30,
        "scans_per_day": 14,
    },
    AssetClass.FUTURES: {
        "active_hours": "18:00-17:00 ET (Sun-Fri)",  # 23h/day
        "interval_minutes": 15,
        "scans_per_day": 92,
    },
    AssetClass.CRYPTO: {
        "active_hours": "24/7",
        "interval_minutes": 10,
        "scans_per_day": 144,
    },
    AssetClass.PREDICTION_MARKETS: {
        "active_hours": "24/7",
        "interval_minutes": 60,
        "scans_per_day": 24,
    },
}
```

### 11.3 Data Provider Dependencies

| Asset Class | Real-time Data | Historical Data | Alternative Data |
|---|---|---|---|
| Options | Polygon.io, Tradier | yfinance | — |
| Futures | Databento (CME L1/L2) | Databento historical | CFTC COT reports |
| Crypto | ccxt (Binance/Coinbase) | ccxt historical | Glassnode, CryptoQuant |
| Prediction | Polymarket CLOB API | Polymarket history | News APIs, polls |

---

## 12. Database Schema Extensions

### 12.1 New Tables

```sql
-- Extend existing database with multi-asset support

-- Generalized trades table (replaces options-only trades)
-- Keep existing trades table as-is for backward compatibility
-- New table for multi-asset trades
CREATE TABLE IF NOT EXISTS multi_asset_trades (
    id TEXT PRIMARY KEY,
    asset_class TEXT NOT NULL,      -- 'options' | 'futures' | 'crypto' | 'prediction_markets'
    source TEXT NOT NULL,           -- 'scanner' | 'manual' | 'paper'
    instrument TEXT NOT NULL,       -- 'SPY', 'ES', 'BTC', market_id
    strategy_type TEXT,             -- 'bull_put_spread', 'mean_reversion', 'funding_arb', etc.
    status TEXT DEFAULT 'open',
    side TEXT,                      -- 'long' | 'short' | 'spread'
    entry_price REAL,
    exit_price REAL,
    quantity REAL,                  -- Contracts, coins, or shares
    leverage REAL DEFAULT 1.0,
    notional_value REAL,           -- Total exposure in USD
    entry_date TEXT,
    exit_date TEXT,
    exit_reason TEXT,
    pnl REAL,
    pnl_pct REAL,
    fees REAL,
    metadata JSON,                 -- Asset-class-specific details
    ml_score REAL,                 -- ML prediction at entry
    unified_score REAL,            -- Unified cross-asset score at entry
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_mat_asset_class ON multi_asset_trades(asset_class);
CREATE INDEX idx_mat_status ON multi_asset_trades(status);
CREATE INDEX idx_mat_instrument ON multi_asset_trades(instrument);
CREATE INDEX idx_mat_entry_date ON multi_asset_trades(entry_date);

-- Portfolio snapshots (periodic state captures)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_value REAL,
    total_pnl REAL,
    total_leverage REAL,
    drawdown_pct REAL,
    positions_json JSON,           -- Full position breakdown
    allocation_json JSON,          -- Per-asset-class allocation
    correlation_json JSON,         -- Current correlation matrix
    risk_metrics_json JSON,        -- VaR, ES, Sharpe, etc.
    created_at TEXT DEFAULT (datetime('now'))
);

-- ML model performance tracking
CREATE TABLE IF NOT EXISTS ml_model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_class TEXT NOT NULL,
    model_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,     -- 'accuracy', 'precision', 'recall', 'sharpe', etc.
    metric_value REAL NOT NULL,
    sample_size INTEGER,
    window TEXT,                   -- '7d', '30d', '90d'
    created_at TEXT DEFAULT (datetime('now'))
);

-- Futures-specific data
CREATE TABLE IF NOT EXISTS futures_term_structure (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_symbol TEXT NOT NULL,     -- 'ES', 'NQ', 'CL'
    snapshot_json JSON NOT NULL,   -- Full term structure at point in time
    contango_backwardation TEXT,   -- 'contango' | 'backwardation' | 'mixed'
    front_back_spread REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Crypto-specific data
CREATE TABLE IF NOT EXISTS crypto_funding_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    funding_rate REAL NOT NULL,
    annualized_rate REAL,
    funding_time TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS crypto_on_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    metrics_json JSON NOT NULL,    -- Full on-chain snapshot
    created_at TEXT DEFAULT (datetime('now'))
);

-- Prediction market tracking
CREATE TABLE IF NOT EXISTS prediction_markets (
    id TEXT PRIMARY KEY,           -- Polymarket market ID
    question TEXT NOT NULL,
    category TEXT,
    resolution_date TEXT,
    current_yes_price REAL,
    model_probability REAL,
    edge REAL,                     -- model_prob - market_price
    liquidity_usd REAL,
    status TEXT DEFAULT 'active',  -- 'active' | 'resolved_yes' | 'resolved_no'
    resolution_value REAL,         -- 1.0 (YES) or 0.0 (NO)
    metadata JSON,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### 12.2 Migration Strategy

```python
# shared/database.py extension

def migrate_v2(path: Optional[str] = None) -> None:
    """Run v2 migration: add multi-asset tables.

    Safe to run multiple times (IF NOT EXISTS).
    Does NOT modify existing tables — backward compatible.
    """
    conn = get_db(path)
    try:
        conn.executescript(V2_SCHEMA)
        conn.commit()
        logger.info("Database migrated to v2 (multi-asset)")
    finally:
        conn.close()
```

---

## 13. Backtesting Framework

### 13.1 Generalized Backtester

```python
# backtest/backtester.py (refactored)

class MultiAssetBacktester:
    """Asset-class-aware backtesting engine.

    Extends existing backtester with:
    - Leverage simulation (futures, crypto)
    - 24/7 market simulation (crypto)
    - Funding rate simulation (crypto perps)
    - Event resolution simulation (prediction markets)
    - Cross-asset portfolio simulation
    """

    def __init__(self, config: Dict):
        self.config = config
        self.asset_backtesters = {
            AssetClass.OPTIONS: OptionsBacktester(config),    # Existing
            AssetClass.FUTURES: FuturesBacktester(config),    # New
            AssetClass.CRYPTO: CryptoBacktester(config),      # New
            AssetClass.PREDICTION_MARKETS: PredictionBacktester(config),  # New
        }

    def run_single_asset(
        self, asset_class: AssetClass, start_date: str, end_date: str
    ) -> Dict:
        """Backtest a single asset class in isolation."""
        ...

    def run_portfolio(self, start_date: str, end_date: str) -> Dict:
        """Backtest full multi-asset portfolio with cross-asset risk management.

        This is the gold standard test — simulates exactly how the live system
        would behave with all asset classes running simultaneously.
        """
        ...
```

### 13.2 Futures Backtester

```python
# backtest/futures_backtester.py

class FuturesBacktester:
    """Futures-specific backtesting with proper handling of:
    - Contract rolls (front month → next month)
    - Margin requirements
    - Tick-based P&L (not percentage-based)
    - Intraday bar data
    - Slippage scaled to volatility
    """

    def simulate_trade(self, signal: Dict, bars: pd.DataFrame) -> Dict:
        """Simulate a single futures trade.

        Accounts for:
        - Entry slippage (1-2 ticks based on volatility)
        - Commission ($2.25/side per contract)
        - Margin requirements (initial + maintenance)
        - Mark-to-market P&L
        """
        ...
```

### 13.3 Crypto Backtester

```python
# backtest/crypto_backtester.py

class CryptoBacktester:
    """Crypto-specific backtesting with:
    - 24/7 market simulation (no gaps)
    - Funding rate payments (every 8 hours for perps)
    - Exchange fee tiers (maker/taker)
    - Liquidation price simulation
    - Basis tracking for spot-perp arb
    """

    def simulate_funding_arb(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Simulate funding rate arbitrage trade.

        Tracks:
        - Spot entry + perp short entry (with slippage)
        - 8-hourly funding payments collected
        - Basis risk (spot-perp price divergence)
        - Margin maintenance on perp leg
        - Total yield net of fees
        """
        ...
```

### 13.4 Prediction Market Backtester

```python
# backtest/prediction_backtester.py

class PredictionBacktester:
    """Prediction market backtesting on historical resolved markets.

    Uses Polymarket's history of resolved markets to test:
    - Model probability accuracy (Brier score)
    - Edge capture (average P&L per unit of edge)
    - Category-specific performance
    - Win rate at various edge thresholds
    """

    def backtest_model(
        self, resolved_markets: List[Dict], model: PredictionMLModel
    ) -> Dict:
        """Run model against historical resolved markets.

        Returns:
            {
                "total_markets": 500,
                "brier_score": 0.18,        # Lower is better (0 = perfect)
                "log_loss": 0.42,
                "calibration_curve": [...],
                "avg_edge_captured": 0.035,  # 3.5 cents per dollar
                "roi": 0.12,                 # 12% return
                "win_rate": 0.58,
                "by_category": {...},
            }
        """
        ...
```

### 13.5 Performance Metrics Extension

```python
# backtest/performance_metrics.py (extended)

def calculate_leverage_adjusted_metrics(trades: List[Dict]) -> Dict:
    """Performance metrics that account for leverage.

    New metrics beyond existing:
    - Leverage-adjusted Sharpe ratio
    - Return on margin (futures)
    - Yield vs risk (funding arb)
    - Information ratio per asset class
    - Maximum leverage used
    - Margin utilization over time
    """
    ...

def calculate_cross_asset_metrics(portfolio_history: pd.DataFrame) -> Dict:
    """Portfolio-level metrics across all asset classes.

    - Diversification ratio
    - Contribution to risk by asset class
    - Correlation stability over time
    - Tail risk (VaR, Expected Shortfall)
    """
    ...
```

---

## 14. Web UI Extensions

### 14.1 New Pages

| Page | Purpose |
|---|---|
| `/portfolio` | Unified dashboard: total P&L, allocation pie chart, correlation heatmap, risk gauges |
| `/futures` | Futures positions, term structure chart, active signals |
| `/crypto` | Crypto positions, funding rates, on-chain metrics dashboard |
| `/predictions` | Active prediction market positions, model vs market probability chart |

### 14.2 New API Routes

```typescript
// web/app/api/portfolio/route.ts
// GET: Full portfolio snapshot across all asset classes

// web/app/api/futures/route.ts
// GET: Futures positions + scan results

// web/app/api/crypto/route.ts
// GET: Crypto positions + funding rates + on-chain

// web/app/api/predictions/route.ts
// GET: Prediction market positions + model probabilities

// web/app/api/scan/route.ts (extended)
// POST body now accepts: { asset_classes: ["options", "futures", "crypto"] }
```

### 14.3 Portfolio Dashboard Components

```
┌─────────────────────────────────────────────────────────┐
│  PILOTAI MULTI-ASSET DASHBOARD                    ⚙ 🔴  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Total Portfolio: $105,230  (+5.23%)   Drawdown: 1.2%  │
│  Active Positions: 10      Leverage: 2.1x              │
│                                                         │
│  ┌─────────────┐  ┌──────────────────────────────────┐ │
│  │  Allocation  │  │  P&L Over Time (stacked area)    │ │
│  │  [PIE CHART] │  │  ───── Options (green)           │ │
│  │  40% Options │  │  ───── Futures (blue)            │ │
│  │  24% Futures │  │  ───── Crypto (orange)           │ │
│  │  17% Crypto  │  │  ───── Predictions (purple)     │ │
│  │  3% PredMkt  │  │                                  │ │
│  └─────────────┘  └──────────────────────────────────┘ │
│                                                         │
│  ┌──────────────────────┐ ┌──────────────────────────┐ │
│  │  Correlation Matrix   │ │  Risk Metrics            │ │
│  │  ES  BTC SPY ETH     │ │  VaR (95%): -$3,200     │ │
│  │ ES [1.0 .3  .9  .2]  │ │  Sharpe (30d): 1.8      │ │
│  │ BTC[.3  1.0 .3  .8]  │ │  Max DD: 4.2%           │ │
│  │ SPY[.9  .3  1.0 .2]  │ │  Win Rate: 72%          │ │
│  │ ETH[.2  .8  .2  1.0] │ │  Profit Factor: 2.3     │ │
│  └──────────────────────┘ └──────────────────────────┘ │
│                                                         │
│  RECENT SIGNALS                                         │
│  ┌──────┬───────┬──────────────┬───────┬──────┬──────┐ │
│  │Asset │Instr  │Signal        │Score  │Edge  │Action│ │
│  ├──────┼───────┼──────────────┼───────┼──────┼──────┤ │
│  │OPT   │SPY    │Bull Put 575/5│82     │1.2%  │EXEC  │ │
│  │FUT   │ES     │Mean Rev Long │76     │3.1%  │EXEC  │ │
│  │CRY   │BTC    │Funding Arb   │71     │12%/yr│EXEC  │ │
│  │PRED  │#12847 │YES @ $0.42   │68     │8.2%  │EXEC  │ │
│  │FUT   │CL     │Momentum Short│55     │1.8%  │ALERT │ │
│  └──────┴───────┴──────────────┴───────┴──────┴──────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 15. Testing Strategy

### 15.1 Test Pyramid

```
                    ┌─────────────────┐
                    │  E2E Tests      │  (5%)
                    │  Full scan →    │
                    │  execution flow │
                    └────────┬────────┘
                 ┌───────────┴───────────┐
                 │  Integration Tests    │  (25%)
                 │  Strategy + ML + DB   │
                 │  Provider + Executor  │
                 └───────────┬───────────┘
          ┌──────────────────┴──────────────────┐
          │  Unit Tests                         │  (70%)
          │  Each module independently          │
          │  Feature engine, models, scoring    │
          │  Risk manager, correlation tracker  │
          └─────────────────────────────────────┘
```

### 15.2 Test Categories

| Category | What | Tools |
|---|---|---|
| Unit tests | Individual functions and classes | pytest, hypothesis |
| Integration tests | Multi-component workflows | pytest, fixtures, mocked APIs |
| Backtest validation | Strategy logic on historical data | Custom backtester |
| ML model tests | Model accuracy, feature drift, calibration | scikit-learn metrics |
| Paper trading tests | End-to-end with paper money | Real API calls (sandbox) |
| Property-based tests | Edge cases in risk calculations | hypothesis |
| Regression tests | Ensure existing options system unaffected | Snapshot testing |

### 15.3 Test Files

```
tests/
├── conftest.py                          # Shared fixtures (extended)
├── test_paper_trader.py                 # Existing
├── test_spread_strategy_full.py         # Existing
├── ...                                  # All existing tests (unchanged)
│
├── test_orchestrator.py                 # NEW: Multi-asset coordination
├── test_unified_scorer.py              # NEW: Cross-asset scoring
├── test_portfolio_manager.py            # NEW: Portfolio risk
├── test_correlation_tracker.py          # NEW: Correlation logic
├── test_leverage_controller.py          # NEW: Leverage management
├── test_drawdown_monitor.py             # NEW: Drawdown circuit breaker
│
├── test_futures_strategy.py             # NEW: Futures signal generation
├── test_futures_provider.py             # NEW: Futures data fetching
├── test_futures_model.py                # NEW: Futures ML model
├── test_futures_features.py             # NEW: Futures feature engineering
├── test_futures_executor.py             # NEW: Futures order execution
├── test_futures_backtester.py           # NEW: Futures backtesting
│
├── test_crypto_strategy.py              # NEW
├── test_crypto_provider.py              # NEW
├── test_crypto_model.py                 # NEW
├── test_crypto_features.py              # NEW
├── test_crypto_executor.py              # NEW
├── test_crypto_backtester.py            # NEW
│
├── test_prediction_strategy.py          # NEW
├── test_polymarket_provider.py          # NEW
├── test_prediction_model.py             # NEW
├── test_prediction_features.py          # NEW
├── test_prediction_backtester.py        # NEW
│
└── fixtures/
    ├── futures_sample_data.json         # NEW
    ├── crypto_sample_data.json          # NEW
    ├── funding_rates_sample.json        # NEW
    ├── polymarket_sample_markets.json   # NEW
    └── on_chain_sample.json             # NEW
```

### 15.4 ML Model Validation

```python
class ModelValidator:
    """Validate ML models before deployment.

    Checks:
    1. Accuracy > minimum threshold (varies by asset class)
    2. Calibration (predicted probabilities match observed frequencies)
    3. No feature leakage (future data in training)
    4. Robustness to perturbation (small input changes don't flip predictions)
    5. Out-of-sample performance (walk-forward cross-validation)
    """

    THRESHOLDS = {
        AssetClass.OPTIONS: {"min_accuracy": 0.60, "max_brier": 0.25},
        AssetClass.FUTURES: {"min_accuracy": 0.45, "max_brier": 0.30},  # 3-class is harder
        AssetClass.CRYPTO: {"min_accuracy": 0.55, "max_brier": 0.28},
        AssetClass.PREDICTION_MARKETS: {"min_accuracy": 0.55, "max_brier": 0.22},
    }
```

---

## 16. Deployment Approach

### 16.1 Rollout Strategy

```
Phase 1 (Weeks 1-2): Infrastructure refactoring
    - Extract BaseStrategy, extend types, add AssetClass registry
    - Extend database schema (v2 migration)
    - All existing tests still pass
    - Options system runs unchanged in production

Phase 2 (Weeks 3-6): Futures module
    - Data provider (Databento or Polygon futures)
    - Feature engineering
    - ML model training (on 5+ years historical data)
    - Backtesting + validation
    - Paper trading for 2+ weeks
    - Web UI page

Phase 3 (Weeks 7-10): Crypto module
    - Exchange integration (ccxt + Binance)
    - On-chain data integration (Glassnode API)
    - Feature engineering
    - ML model training
    - Funding rate arbitrage backtesting
    - Paper trading for 2+ weeks
    - Web UI page

Phase 4 (Weeks 11-13): Prediction markets module
    - Polymarket API integration
    - Historical resolved markets dataset
    - Probability estimation model training
    - Backtesting on resolved markets
    - Paper trading for 1+ week
    - Web UI page

Phase 5 (Weeks 14-16): Portfolio integration
    - Portfolio manager (cross-asset risk)
    - Correlation tracker
    - Unified scoring calibration
    - Full multi-asset backtesting
    - Portfolio dashboard
    - Comprehensive testing

Phase 6 (Weeks 17-18): Production readiness
    - Load testing
    - Monitoring + alerting
    - Documentation
    - Gradual rollout (enable one asset class at a time)
```

### 16.2 Docker Extension

```dockerfile
# Dockerfile (extended)
FROM python:3.11-slim

# Install additional dependencies for multi-asset
RUN pip install ccxt databento glassnode-api

# ... rest of existing Dockerfile ...

# Extended health check covers all asset class schedulers
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "from shared.database import get_db; get_db().execute('SELECT 1')" || exit 1
```

### 16.3 Configuration for Gradual Rollout

```yaml
# config.yaml — enable asset classes one at a time
asset_classes:
  options:
    enabled: true       # Already in production
  futures:
    enabled: false      # Flip to true when ready
  crypto:
    enabled: false      # Flip to true when ready
  prediction_markets:
    enabled: false      # Flip to true when ready
```

---

## 17. Implementation Timeline

```
Week  1-2:  ████████░░░░░░░░░░░░ Phase 1: Core Infrastructure
Week  3-6:  ░░░░████████████░░░░ Phase 2: Futures Module
Week  7-10: ░░░░░░░░░░██████████ Phase 3: Crypto Module
Week 11-13: ░░░░░░░░░░░░░░██████ Phase 4: Prediction Markets
Week 14-16: ░░░░░░░░░░░░░░░░████ Phase 5: Portfolio Integration
Week 17-18: ░░░░░░░░░░░░░░░░░░██ Phase 6: Production Readiness
```

### New Dependencies

```
# requirements.txt additions

# Futures data
databento>=0.30.0              # CME market data

# Crypto
ccxt>=4.0.0                    # Unified exchange library
# glassnode-api (if available, else raw HTTP)

# ML extensions
lightgbm>=4.0.0                # Ensemble diversity for crypto model

# Async support (for 24/7 scanning)
aiohttp>=3.9.0                 # Async HTTP for concurrent data fetching
asyncio                        # Standard library

# Prediction markets
web3>=6.0.0                    # Ethereum/Polygon interaction (Polymarket)
py-clob-client>=0.10.0         # Polymarket CLOB SDK
```

### Risk Mitigation

| Risk | Mitigation |
|---|---|
| Breaking existing options system | Feature flags per asset class; existing tests as regression suite |
| Bad ML model deployed | Walk-forward validation; paper trading gate; automatic fallback to rules-based |
| API rate limits | Circuit breakers on all providers; request caching; respect rate limits |
| Excessive leverage | Leverage controller with drawdown-based scaling; hard caps in config |
| Correlated losses across assets | Correlation tracker; max correlation overlap check before new trades |
| Polymarket API changes | Abstraction layer; version pinning; monitoring for breaking changes |
| Data quality issues | Schema validation on ingestion; anomaly detection on features; stale data alerts |

---

## Summary

This architecture extends PilotAI from a single-asset options system into a four-asset-class ML-powered trading platform while:

1. **Preserving** the production options system (zero-downtime migration)
2. **Reusing** existing patterns (DataProvider protocol, MLPipeline, FeatureEngine, SQLite WAL, circuit breaker)
3. **Adding** asset-class-specific strategies, features, and ML models
4. **Unifying** scoring across asset classes for portfolio-level optimization
5. **Managing** risk at both the position and portfolio level with leverage-aware controls

The system is designed to be built incrementally (futures → crypto → prediction markets) with each phase independently valuable and testable.
