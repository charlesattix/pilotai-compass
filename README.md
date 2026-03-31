# 🧭 COMPASS — Complete Portfolio Analysis & Strategy System

A comprehensive quantitative trading platform with 172 modules, ~8,000 tests, and full pipeline coverage from data ingestion to execution analytics.

## Architecture

```
Data → Signals → Models → Sizing → Risk → Execution → Analytics
```

### 🔌 Data Layer
Data pipeline manager, quality monitoring, feature store (SQLite-backed), training data collectors.

### 🧠 Signal Generation (~30+ modules)
Cross-asset signals, order flow, institutional trade flow (VPIN), sentiment, event calendar, macro signals, multi-timeframe aggregation, signal decay analysis, signal quality scoring, alpha research framework.

### 🎯 ML Models & Prediction
Ensemble models (XGBoost, LightGBM, stacking), walk-forward validation, feature importance + pruning, regime detection (HMM), model monitoring with drift detection.

### 📏 Position Sizing & Portfolio
Fractional Kelly, drawdown-reactive, regime-adaptive, correlation-aware sizing. Portfolio optimization (Markowitz, Black-Litterman, risk parity, min-CVaR). MC optimizer with efficient frontier.

### 🛡️ Risk Management
Dynamic risk limits, kill switch, stress testing (GFC/COVID/Flash Crash), VaR/CVaR with EVT tail fitting, margin monitoring, risk budgeting, overnight risk.

### ⚡ Execution
TWAP/VWAP/IS/Iceberg algorithms, smart order routing, Almgren-Chriss market impact, Avellaneda-Stoikov market-making sim, RL execution agent, execution analytics.

### 📊 Analytics & Reporting
Performance attribution (Brinson + factor), experiment ranking (S/A/B/C/D/F tiers), North Star gap analysis, backtest reality checking, 20+ HTML dashboards.

### 🔧 Infrastructure
Pipeline validator, deploy checklist, test health analyzer, production monitoring with alerts, module auditor, system integration engine.

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Stats

- **172** compass modules
- **232** test files
- **~8,000** tests passing
- **0** failures

## Built By

Maximus ⚡ — 5 parallel Claude Code sessions, relentless experimentation.
