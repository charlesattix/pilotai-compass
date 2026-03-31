# 🔬 Experiments Directory

Each experiment lives in its own folder with a standardized structure.

## Structure

```
experiments/
├── EXP-XXX-max/
│   ├── THESIS.md          # Hypothesis, rationale, expected outcome
│   ├── CONFIG.yaml        # Strategy parameters, entry/exit rules
│   ├── backtest.py        # Backtest implementation
│   ├── results/           # Backtest output (metrics, trades, charts)
│   │   ├── summary.json   # Key metrics (Sharpe, returns, DD, etc.)
│   │   └── report.html    # Visual report
│   ├── analysis.md        # Post-backtest analysis & lessons learned
│   └── STATUS.md          # Current status (backtest/paper/live/killed)
```

## Experiment Lifecycle

1. **THESIS** — Write hypothesis before any code
2. **BUILD** — Implement strategy + backtest
3. **BACKTEST** — Run against 2020-2025 historical data
4. **VALIDATE** — Walk-forward out-of-sample validation
5. **ANALYZE** — Document results, compare to North Star targets
6. **DECISION** — Promote to paper trading OR kill with documented reason

## North Star Targets

| Metric | Target |
|--------|--------|
| Annual Returns | 100%+ |
| Max Drawdown | ≤12% |
| Sharpe Ratio | 6.0+ |
| AUM Capacity | Billions |

## Active Experiments

| ID | Name | CC Session | Status |
|----|------|-----------|--------|
| EXP-702-max | Multi-Asset Portfolio | cc1 | Backtesting |
| EXP-710-max | Aggressive ML Filter | cc2 | Backtesting |
| EXP-720-max | Dynamic Regime Sizing | cc3 | Backtesting |
| EXP-730-max | Short DTE Theta | cc4 | Backtesting |
| EXP-740-max | Volatility Harvesting | cc5 | Backtesting |
