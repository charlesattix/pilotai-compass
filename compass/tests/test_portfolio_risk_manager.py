"""Unit tests for compass/portfolio_risk_manager.py (EXP-1890)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from compass.regime import Regime
from compass.portfolio_risk_manager import (
    AllocationLimiter,
    CircuitState,
    CorrelationAlert,
    CorrelationMonitor,
    CrossStrategySizer,
    DrawdownCircuitBreaker,
    LeverageGovernor,
    PortfolioRiskManager,
    REGIME_LEVERAGE,
    STRESS_REGIMES,
)


# ───────────────────────────────────────────────────────────────────────────
# Fixtures: deterministic, no randomness
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def returns_3strats() -> pd.DataFrame:
    """120 days of deterministic returns for 3 strategies, distinct vols."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    # Use sin/cos with different amplitudes → reproducible distinct vols
    a = 0.003 * np.sin(np.arange(120) * 0.4)
    b = 0.006 * np.sin(np.arange(120) * 0.3 + 1)
    c = 0.012 * np.cos(np.arange(120) * 0.5)
    return pd.DataFrame({"alpha": a, "beta": b, "gamma": c}, index=dates)


@pytest.fixture
def returns_correlated() -> pd.DataFrame:
    """Two strategies that move identically + a third independent one."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    base = 0.01 * np.sin(np.arange(60) * 0.5)
    return pd.DataFrame(
        {
            "stratA": base,
            "stratB": base * 0.99 + 1e-5,         # ρ ≈ 1
            "stratC": 0.005 * np.cos(np.arange(60) * 0.7),
        },
        index=dates,
    )


# ───────────────────────────────────────────────────────────────────────────
# 1. CrossStrategySizer
# ───────────────────────────────────────────────────────────────────────────


def test_sizer_returns_weights_summing_to_one(returns_3strats):
    s = CrossStrategySizer(vol_lookback=60)
    w = s.size(returns_3strats)
    assert set(w.keys()) == {"alpha", "beta", "gamma"}
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    assert all(v >= 0 for v in w.values())


def test_sizer_inverse_vol_orders_weights(returns_3strats):
    s = CrossStrategySizer()
    w = s.size(returns_3strats)
    # gamma has the largest amplitude → highest vol → smallest weight
    assert w["gamma"] < w["beta"] < w["alpha"]


def test_sizer_kelly_caps_weights(returns_3strats):
    s = CrossStrategySizer(use_kelly=True, kelly_cap=0.20, kelly_fraction=0.5)
    w = s.size(returns_3strats)
    assert all(v <= 0.20 + 1e-9 for v in w.values())
    assert sum(w.values()) <= 1.0 + 1e-9


def test_sizer_handles_empty_input():
    assert CrossStrategySizer().size(pd.DataFrame()) == {}


def test_sizer_short_history_uses_equal_weight():
    df = pd.DataFrame({"a": [0.01, 0.02], "b": [0.0, -0.01]})
    w = CrossStrategySizer(vol_lookback=60).size(df)
    assert math.isclose(w["a"], 0.5) and math.isclose(w["b"], 0.5)


# ───────────────────────────────────────────────────────────────────────────
# 2. CorrelationMonitor
# ───────────────────────────────────────────────────────────────────────────


def test_monitor_silent_in_calm_regime(returns_correlated):
    m = CorrelationMonitor(window=30, threshold=0.5)
    assert m.check(returns_correlated, Regime.BULL) == []  # not stress regime


def test_monitor_alerts_in_stress_regime(returns_correlated):
    m = CorrelationMonitor(window=30, threshold=0.5)
    alerts = m.check(returns_correlated, Regime.HIGH_VOL)
    assert any(p == ("stratA", "stratB") for p, _ in alerts)
    assert all(abs(rho) >= 0.5 for _, rho in alerts)


def test_monitor_check_or_raise(returns_correlated):
    m = CorrelationMonitor(window=30, threshold=0.5)
    with pytest.raises(CorrelationAlert):
        m.check_or_raise(returns_correlated, Regime.CRASH)


def test_monitor_always_alert_overrides_regime(returns_correlated):
    m = CorrelationMonitor(window=30, threshold=0.5, always_alert=True)
    assert m.check(returns_correlated, Regime.BULL)  # ignores regime gating


def test_monitor_skips_when_too_few_columns():
    df = pd.DataFrame({"only": [0.01] * 60})
    assert CorrelationMonitor().check(df, Regime.CRASH) == []


# ───────────────────────────────────────────────────────────────────────────
# 3. DrawdownCircuitBreaker
# ───────────────────────────────────────────────────────────────────────────


def test_circuit_starts_normal():
    cb = DrawdownCircuitBreaker()
    assert cb.update(100_000) == CircuitState.NORMAL
    assert cb.leverage_multiplier() == 1.0


def test_circuit_warn_at_soft_trip():
    cb = DrawdownCircuitBreaker(soft_pct=0.10, hard_pct=0.12)
    cb.update(100_000)
    state = cb.update(89_500)  # 10.5% DD
    assert state == CircuitState.WARN
    assert cb.leverage_multiplier() == 0.5


def test_circuit_halt_at_hard_trip_and_sticky():
    cb = DrawdownCircuitBreaker(soft_pct=0.10, hard_pct=0.12)
    cb.update(100_000)
    cb.update(87_000)  # 13% DD → HALT
    assert cb.state == CircuitState.HALT
    assert cb.leverage_multiplier() == 0.0
    # Recovery does NOT auto-clear HALT — must call reset()
    cb.update(100_000)
    assert cb.state == CircuitState.HALT
    cb.reset()
    cb.update(100_000)
    assert cb.state == CircuitState.NORMAL


def test_circuit_recovers_from_warn_when_dd_shrinks():
    cb = DrawdownCircuitBreaker(soft_pct=0.10, hard_pct=0.12)
    cb.update(100_000)
    cb.update(89_500)
    assert cb.state == CircuitState.WARN
    cb.update(95_000)        # 5% DD
    assert cb.state == CircuitState.NORMAL


def test_circuit_current_dd():
    cb = DrawdownCircuitBreaker()
    cb.update(100_000)
    cb.update(80_000)        # 20%
    assert math.isclose(cb.current_dd(80_000), 0.20, abs_tol=1e-9)


# ───────────────────────────────────────────────────────────────────────────
# 4. AllocationLimiter
# ───────────────────────────────────────────────────────────────────────────


def test_limiter_caps_max_weight():
    lim = AllocationLimiter(max_per_strategy=0.30, min_per_strategy=0.0)
    out = lim.apply({"a": 0.6, "b": 0.25})
    assert out["a"] == 0.30   # capped
    assert out["b"] == 0.25   # under cap, untouched
    assert sum(out.values()) <= 1.0 + 1e-9


def test_limiter_drops_min_weight():
    lim = AllocationLimiter(max_per_strategy=1.0, min_per_strategy=0.10)
    out = lim.apply({"a": 0.5, "b": 0.05})
    assert out["b"] == 0.0
    assert out["a"] == 0.5


def test_limiter_renormalises_when_total_above_one():
    lim = AllocationLimiter(max_per_strategy=1.0, min_per_strategy=0.0)
    out = lim.apply({"a": 0.7, "b": 0.7})
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)


def test_limiter_needs_rebalance():
    lim = AllocationLimiter(rebalance_threshold=0.10)
    assert lim.needs_rebalance({"a": 0.5, "b": 0.5}, {"a": 0.35, "b": 0.65})
    assert not lim.needs_rebalance({"a": 0.5}, {"a": 0.55})


# ───────────────────────────────────────────────────────────────────────────
# 5. LeverageGovernor
# ───────────────────────────────────────────────────────────────────────────


def test_governor_regime_table_in_1x_3x_range():
    g = LeverageGovernor()
    for r in [Regime.BULL, Regime.LOW_VOL, Regime.HIGH_VOL, Regime.BEAR, Regime.CRASH]:
        assert 1.0 <= g.base_leverage(r) <= 3.0


def test_governor_bull_higher_than_crash():
    g = LeverageGovernor()
    assert g.base_leverage(Regime.BULL) > g.base_leverage(Regime.CRASH)
    assert math.isclose(g.base_leverage(Regime.BULL), 3.0, abs_tol=1e-9)
    assert math.isclose(g.base_leverage(Regime.CRASH), 1.0, abs_tol=1e-9)


def test_governor_realised_vol_clamp():
    # Very high realised vol should pull leverage down toward min_lev.
    s = pd.Series(np.linspace(-0.05, 0.05, 30))   # huge swings
    g = LeverageGovernor(target_vol=0.05)
    lev = g.leverage(Regime.BULL, s)
    assert lev <= 3.0
    assert lev >= 1.0


def test_governor_clamped_to_min_max():
    g = LeverageGovernor(min_lev=1.0, max_lev=2.0,
                         regime_table={Regime.BULL: 5.0, Regime.CRASH: 0.1})
    assert g.leverage(Regime.BULL, None) == 2.0
    assert g.leverage(Regime.CRASH, None) == 1.0


# ───────────────────────────────────────────────────────────────────────────
# Composition: PortfolioRiskManager
# ───────────────────────────────────────────────────────────────────────────


def test_manager_normal_decision(returns_3strats):
    mgr = PortfolioRiskManager()
    d = mgr.make_decision(returns_3strats, Regime.BULL, equity=100_000)
    assert d.state == CircuitState.NORMAL
    assert d.regime == Regime.BULL
    assert d.leverage > 1.0
    # weights non-empty, all in [0, max_cap], total ≤ 1
    assert d.weights
    assert all(0.0 <= w <= 0.40 + 1e-9 for w in d.weights.values())
    assert sum(d.weights.values()) <= 1.0 + 1e-9
    assert d.correlation_alerts == []


def test_manager_warn_state_deleverages(returns_3strats):
    mgr = PortfolioRiskManager()
    mgr.breaker.update(100_000)             # set HWM
    d = mgr.make_decision(returns_3strats, Regime.BULL, equity=89_000)
    assert d.state == CircuitState.WARN
    # Leverage halved by circuit
    assert d.leverage <= REGIME_LEVERAGE[Regime.BULL] * 0.5 + 1e-9
    assert any("circuit" in n for n in d.notes)


def test_manager_halt_flattens(returns_3strats):
    mgr = PortfolioRiskManager()
    mgr.breaker.update(100_000)
    d = mgr.make_decision(returns_3strats, Regime.BEAR, equity=85_000)
    assert d.state == CircuitState.HALT
    assert d.leverage == 0.0
    assert all(w == 0.0 for w in d.weights.values())


def test_manager_correlation_alert_in_stress(returns_correlated):
    mgr = PortfolioRiskManager()
    d = mgr.make_decision(returns_correlated, Regime.HIGH_VOL, equity=100_000)
    assert d.correlation_alerts
    assert any("correlation alert" in n for n in d.notes)


def test_manager_decision_serialises(returns_3strats):
    mgr = PortfolioRiskManager()
    d = mgr.make_decision(returns_3strats, Regime.LOW_VOL, equity=100_000)
    payload = d.to_dict()
    assert "weights" in payload and "leverage" in payload and "state" in payload
    assert payload["regime"] == "low_vol"


def test_manager_rebalance_flag(returns_3strats):
    mgr = PortfolioRiskManager(limiter=AllocationLimiter(rebalance_threshold=0.05))
    # Provide a current_weights set far from any sensible target
    d = mgr.make_decision(returns_3strats, Regime.BULL, equity=100_000,
                          current_weights={"alpha": 0.0, "beta": 0.0, "gamma": 1.0})
    assert d.rebalance is True


def test_stress_regimes_set():
    assert Regime.BEAR in STRESS_REGIMES
    assert Regime.HIGH_VOL in STRESS_REGIMES
    assert Regime.CRASH in STRESS_REGIMES
    assert Regime.BULL not in STRESS_REGIMES
