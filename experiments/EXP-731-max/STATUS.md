# Status: ANALYSIS COMPLETE — FAILED

**Started:** 2026-03-31
**Parent:** EXP-730-max
**Phase:** Backtest complete, analysis written

## Results Summary
- **Total Return:** -22.67% — **FAILED**
- **CAGR:** -4.20% — **FAILED**
- **Sharpe:** -3.11 — **FAILED**
- **Max DD:** 24.31% — **FAILED** (>15%)
- **Win Rate:** 46.1% — **FAILED** (<70%)
- **Trades:** 347 (58/yr) — partial success (target 150+/yr)

## Key Finding
Removing IV rank filter destroys the edge entirely. The alpha is in **selective entry during elevated IV**, not in trade frequency. Low-vol regime alone lost $16,000 (71% of all losses). Slippage on $1-2 spreads consumed $43,566 (192% of gross loss).

## Verdict
**DISPROVED:** High-frequency trading without IV selection is not viable for short DTE credit spreads. IV rank filter is the strategy's core alpha source.

## Timeline
- 2026-03-31: Thesis written, forked from EXP-730
- 2026-03-31: Backtest complete (347 trades, -22.67% return)
- 2026-03-31: Analysis written — definitively disproves frequency hypothesis
