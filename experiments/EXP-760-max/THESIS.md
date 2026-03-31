# EXP-760-max: ML Filter + Dynamic Regime Sizing

## Hypothesis

Combining EXP-710's ML signal filter (P≥0.60, best total P&L $174K) with
dynamic regime-based position sizing will improve risk-adjusted returns
further.  The ML filter selects *which* trades to take; regime sizing
controls *how much* capital to allocate per trade based on market
conditions.

## Rationale

- EXP-710 at P≥0.60: 192 trades, 85.4% WR, Sharpe 9.72, DD -5.0%, P&L +$174K
- Bull regimes historically have higher win rates and wider margins
- Bear and high-vol regimes produce larger drawdowns per trade
- Sizing up in favorable regimes and down in hostile ones should:
  - Increase total returns (more capital deployed when edge is strongest)
  - Reduce drawdown (less capital at risk in dangerous regimes)
  - Improve Sharpe (better risk allocation)

## Strategy

1. Train walk-forward XGBoost + RF ensemble (same as EXP-710)
2. Filter trades at P(win) ≥ 0.60
3. Apply regime-dependent position multipliers to PnL:
   - Bull: 2.0x (full aggression, edge is strongest)
   - Neutral/default: 1.0x (baseline)
   - Bear: 0.5x (defensive, high uncertainty)
   - High-vol / crash: 0.25x (minimal exposure)
4. Sweep multiplier configurations to find optimal regime weights

## Multiplier Sweep

| Config   | Bull | Neutral | Bear | High-Vol |
|----------|------|---------|------|----------|
| Baseline | 1.0  | 1.0     | 1.0  | 1.0      |
| Moderate | 1.5  | 1.0     | 0.5  | 0.25     |
| Aggressive | 2.0 | 1.0    | 0.5  | 0.25     |
| Very Aggressive | 2.5 | 1.0 | 0.3 | 0.1     |
| Conservative | 1.25 | 1.0  | 0.75 | 0.5     |

## Success Criteria

- Sharpe improvement ≥ 20% over EXP-710 at P≥0.60 (target: Sharpe > 11.7)
- Total P&L improvement ≥ 10% (target: > $192K)
- Max DD does not worsen significantly (target: < 8%)
- Win rate maintained ≥ 84%

## Risks

- Bull sizing amplifies losses when model is wrong in bull markets
- Regime labels may be noisy or lagged
- Multiplier optimization is another degree of freedom (overfit risk)
- Regime distribution is uneven (bull dominates) — may just be a leverage play
