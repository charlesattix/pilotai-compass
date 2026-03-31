"""Tests for compass/adaptive_stops.py — regime-aware stop loss optimizer.

Covers:
  - Stop strategy functions: fixed, ATR, VIX, time-decay, regime, trailing
  - simulate_stop: triggered/not triggered, premature detection
  - backtest_strategy: aggregate metrics
  - AdaptiveStopOptimizer: optimize, regime optimals, premature analysis
  - HTML report generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.adaptive_stops import (
    STOP_STRATEGIES,
    AdaptiveStopOptimizer,
    RegimeOptimal,
    StopResult,
    StrategyResult,
    atr_stop,
    backtest_strategy,
    fixed_stop,
    regime_stop,
    simulate_stop,
    time_decay_stop,
    trailing_stop,
    vix_stop,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _trade(
    pnl=100.0, net_credit=1.5, dte_at_entry=30, hold_days=10,
    contracts=2, spread_width=5.0, vix=20, realized_vol_20d=18,
    regime="bull", exit_reason="close_profit_target", return_pct=10.0,
    **kw,
):
    d = {
        "pnl": pnl, "net_credit": net_credit, "dte_at_entry": dte_at_entry,
        "hold_days": hold_days, "contracts": contracts, "spread_width": spread_width,
        "vix": vix, "realized_vol_20d": realized_vol_20d, "regime": regime,
        "exit_reason": exit_reason, "return_pct": return_pct,
    }
    d.update(kw)
    return pd.Series(d)


def _make_trades(n=30, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        pnl = rng.normal(30, 200)
        rows.append({
            "entry_date": f"2024-{1+i%12:02d}-{1+i%28:02d}",
            "exit_date": f"2024-{1+i%12:02d}-{5+i%24:02d}",
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / 500, 2),
            "win": 1 if pnl > 0 else 0,
            "net_credit": round(rng.uniform(0.5, 3.0), 4),
            "spread_width": 5.0,
            "max_loss_per_unit": round(rng.uniform(3, 5), 4),
            "contracts": rng.randint(1, 5),
            "dte_at_entry": rng.randint(10, 50),
            "hold_days": rng.randint(1, 20),
            "vix": round(rng.uniform(12, 40), 2),
            "realized_vol_20d": round(rng.uniform(10, 35), 2),
            "regime": rng.choice(["bull", "bear", "high_vol", "neutral"]),
            "exit_reason": rng.choice(["close_profit_target", "close_stop_loss", "close_expiration"]),
            "strategy_type": "CS",
            "day_of_week": rng.randint(0, 5),
        })
    return pd.DataFrame(rows)


# ── Stop strategy functions ──────────────────────────────────────────────


class TestFixedStop:
    def test_default(self):
        assert fixed_stop(_trade()) == 3.5

    def test_custom(self):
        assert fixed_stop(_trade(), multiplier=2.0) == 2.0


class TestAtrStop:
    def test_normal_vol(self):
        t = _trade(realized_vol_20d=15)
        m = atr_stop(t)
        assert m == pytest.approx(2.5, abs=0.1)

    def test_high_vol_wider(self):
        t_high = _trade(realized_vol_20d=40)
        t_low = _trade(realized_vol_20d=10)
        assert atr_stop(t_high) > atr_stop(t_low)

    def test_clamped(self):
        assert 1.5 <= atr_stop(_trade(realized_vol_20d=100)) <= 6.0
        assert 1.5 <= atr_stop(_trade(realized_vol_20d=1)) <= 6.0


class TestVixStop:
    def test_normal_vix(self):
        m = vix_stop(_trade(vix=20))
        assert m == pytest.approx(2.5, abs=0.1)

    def test_high_vix_wider(self):
        assert vix_stop(_trade(vix=40)) > vix_stop(_trade(vix=15))

    def test_clamped(self):
        assert 1.5 <= vix_stop(_trade(vix=80)) <= 6.0


class TestTimeDecayStop:
    def test_early_in_trade(self):
        m = time_decay_stop(_trade(dte_at_entry=30, hold_days=3))
        assert m > 3.5  # still wide early

    def test_late_in_trade(self):
        m = time_decay_stop(_trade(dte_at_entry=30, hold_days=28))
        assert m < 2.5  # tightened near expiry

    def test_monotonically_tightens(self):
        prev = 10.0
        for hold in range(1, 30):
            m = time_decay_stop(_trade(dte_at_entry=30, hold_days=hold))
            assert m <= prev + 0.01  # non-increasing
            prev = m


class TestRegimeStop:
    def test_bull(self):
        assert regime_stop(_trade(regime="bull")) == 3.0

    def test_crash_tighter(self):
        assert regime_stop(_trade(regime="crash")) < regime_stop(_trade(regime="bull"))

    def test_custom_multipliers(self):
        m = regime_stop(_trade(regime="bull"), regime_multipliers={"bull": 5.0})
        assert m == 5.0


class TestTrailingStop:
    def test_bull_full_trail(self):
        m = trailing_stop(_trade(regime="bull"))
        assert m == pytest.approx(2.0, abs=0.1)

    def test_crash_tighter_trail(self):
        assert trailing_stop(_trade(regime="crash")) < trailing_stop(_trade(regime="bull"))


# ── simulate_stop ────────────────────────────────────────────────────────


class TestSimulateStop:
    def test_profitable_trade_not_stopped(self):
        t = _trade(pnl=200, exit_reason="close_profit_target", return_pct=15)
        r = simulate_stop(t, fixed_stop, {"multiplier": 3.5})
        assert not r.triggered
        assert r.pnl_at_stop == 200

    def test_stopped_trade_triggered(self):
        # Actual loss $500. Stop at 2.0× on $1.5 credit × 1 contract = $300.
        # $300 < $500*0.95=$475 → tighter stop → triggered.
        t = _trade(pnl=-500, exit_reason="close_stop_loss", net_credit=1.5,
                   contracts=1, return_pct=-50)
        r = simulate_stop(t, fixed_stop, {"multiplier": 2.0})
        assert r.triggered

    def test_wider_stop_prevents_trigger(self):
        # Actual loss = $500, wider stop at 5× would be $1500 → not triggered
        t = _trade(pnl=-500, exit_reason="close_stop_loss", net_credit=1.5,
                   contracts=1, return_pct=-33)
        r = simulate_stop(t, fixed_stop, {"multiplier": 5.0})
        # 5.0 × 1.5 × 1 × 100 = $750 > $500 → wider, not triggered
        assert not r.triggered

    def test_premature_detection(self):
        # Trade was stopped but actual P&L was positive (unusual edge case)
        t = _trade(pnl=50, exit_reason="close_stop_loss", return_pct=5)
        r = simulate_stop(t, fixed_stop, {"multiplier": 2.0})
        if r.triggered:
            assert r.premature


# ── backtest_strategy ────────────────────────────────────────────────────


class TestBacktestStrategy:
    def test_returns_strategy_result(self):
        trades = _make_trades(30)
        result = backtest_strategy(trades, "test", fixed_stop, {"multiplier": 3.5})
        assert isinstance(result, StrategyResult)
        assert result.n_trades == 30

    def test_win_rate_in_range(self):
        trades = _make_trades(30)
        result = backtest_strategy(trades, "test", fixed_stop, {"multiplier": 3.5})
        assert 0 <= result.win_rate <= 1

    def test_stop_rate_in_range(self):
        trades = _make_trades(30)
        result = backtest_strategy(trades, "test", fixed_stop, {"multiplier": 3.5})
        assert 0 <= result.stop_rate <= 1

    def test_empty_trades(self):
        result = backtest_strategy(pd.DataFrame(), "test", fixed_stop, {"multiplier": 3.5})
        assert result.n_trades == 0


# ── AdaptiveStopOptimizer ───────────────────────────────────────────────


class TestAdaptiveStopOptimizer:
    def test_optimize_runs(self):
        trades = _make_trades(50)
        opt = AdaptiveStopOptimizer(trades)
        results = opt.optimize()
        assert "global" in results
        assert "by_regime" in results
        assert "regime_optimals" in results

    def test_all_strategies_tested(self):
        trades = _make_trades(50)
        opt = AdaptiveStopOptimizer(trades)
        opt.optimize()
        assert len(opt.strategy_results) == len(STOP_STRATEGIES)

    def test_regime_optimals_found(self):
        trades = _make_trades(50)
        opt = AdaptiveStopOptimizer(trades)
        opt.optimize()
        assert len(opt.regime_optimals) > 0
        for r in opt.regime_optimals:
            assert isinstance(r, RegimeOptimal)
            assert r.best_strategy in STOP_STRATEGIES

    def test_premature_analysis(self):
        trades = _make_trades(50)
        opt = AdaptiveStopOptimizer(trades)
        results = opt.optimize()
        premature = results["premature_analysis"]
        assert "n_stopped" in premature
        assert "by_regime" in premature

    def test_from_csv(self, tmp_path):
        csv = tmp_path / "trades.csv"
        _make_trades(30).to_csv(csv, index=False)
        opt = AdaptiveStopOptimizer.from_csv(str(csv))
        assert len(opt.trades) == 30

    def test_generate_report(self, tmp_path):
        trades = _make_trades(50)
        opt = AdaptiveStopOptimizer(trades)
        path = opt.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Strategy Comparison" in content
        assert "Regime" in content
        assert "data:image/png;base64," in content

    def test_report_no_external(self, tmp_path):
        trades = _make_trades(40)
        opt = AdaptiveStopOptimizer(trades)
        path = opt.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "http://" not in content
        assert "https://" not in content
