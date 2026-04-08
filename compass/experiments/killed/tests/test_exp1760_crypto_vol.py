"""Smoke and unit tests for compass/exp1760_crypto_vol.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from compass.experiments.killed.exp1760_crypto_vol import (
    REGIME_CALM,
    REGIME_CRISIS,
    REGIME_ELEVATED,
    REGIME_NORMAL,
    REGIME_SIZING,
    bs_put,
    classify_crypto_regime,
    compute_btc_drawdown,
    compute_btc_rvol,
    compute_metrics,
    estimate_iv_from_btc,
    put_credit_spread,
    run_backtest,
    trades_to_daily_returns,
)


def test_bs_put_atm_positive():
    p = bs_put(100, 100, 0.25, 0.30)
    assert p > 0
    assert p < 100  # bounded by strike


def test_bs_put_deep_itm_near_discounted_intrinsic():
    # Deep ITM put approaches K*exp(-rT) - S, slightly less than K-S
    p = bs_put(50, 100, 0.25, 0.30, r=0.045)
    discounted_intrinsic = 100 * math.exp(-0.045 * 0.25) - 50
    assert p >= discounted_intrinsic - 0.5
    assert p < 100 - 50 + 0.5


def test_bs_put_zero_T_returns_intrinsic():
    assert bs_put(80, 100, 0.0, 0.30) == 20.0


def test_credit_spread_positive_for_otm():
    # short 95 / long 92, spot 100, 30d, vol 60%
    val = put_credit_spread(100, 95, 92, 30 / 365.0, 0.60)
    assert val > 0
    assert val < 3.0  # bounded by spread width


def test_classify_regime_crisis_via_rvol():
    assert classify_crypto_regime(rvol=1.0, dd=0.0, gbtc_dev=0.0) == REGIME_CRISIS


def test_classify_regime_crisis_via_drawdown():
    assert classify_crypto_regime(rvol=0.50, dd=-0.30, gbtc_dev=0.0) == REGIME_CRISIS


def test_classify_regime_crisis_via_gbtc():
    assert classify_crypto_regime(rvol=0.30, dd=-0.05, gbtc_dev=-0.20) == REGIME_CRISIS


def test_classify_regime_elevated():
    assert classify_crypto_regime(rvol=0.70, dd=-0.05, gbtc_dev=0.0) == REGIME_ELEVATED


def test_classify_regime_calm():
    assert classify_crypto_regime(rvol=0.30, dd=-0.03, gbtc_dev=0.0) == REGIME_CALM


def test_classify_regime_normal_default():
    assert classify_crypto_regime(rvol=0.50, dd=-0.05, gbtc_dev=0.0) == REGIME_NORMAL


def test_regime_sizing_table_complete():
    for r in [REGIME_CALM, REGIME_NORMAL, REGIME_ELEVATED, REGIME_CRISIS]:
        assert r in REGIME_SIZING
    assert REGIME_SIZING[REGIME_CRISIS] == 0.0
    assert REGIME_SIZING[REGIME_CALM] > REGIME_SIZING[REGIME_NORMAL]


def test_btc_rvol_positive():
    idx = pd.bdate_range("2024-01-01", periods=120)
    np.random.seed(7)
    px = pd.Series(60000 * np.exp(np.cumsum(np.random.normal(0, 0.02, 120))), index=idx)
    rvol = compute_btc_rvol(px, 30)
    assert (rvol.dropna() >= 0).all()
    assert 0.10 < float(rvol.dropna().mean()) < 1.0


def test_btc_drawdown_non_positive():
    idx = pd.bdate_range("2024-01-01", periods=120)
    px = pd.Series(np.linspace(100, 60, 120), index=idx)
    dd = compute_btc_drawdown(px, 60)
    assert (dd.dropna() <= 1e-9).all()


def test_estimate_iv_from_btc_reasonable():
    idx = pd.bdate_range("2024-01-01", periods=200)
    np.random.seed(11)
    px = pd.Series(60000 * np.exp(np.cumsum(np.random.normal(0, 0.025, 200))), index=idx)
    iv = estimate_iv_from_btc(px, mult=1.15, window=20)
    mean_iv = float(iv.mean())
    assert 0.20 < mean_iv < 1.50


def test_run_backtest_smoke_with_synthetic_path():
    """Use a synthetic but DETERMINISTIC trending IBIT-like path for the
    smoke test only — never used for any reported metric."""
    idx = pd.bdate_range("2024-01-15", periods=200)
    spot = pd.Series(40 + 0.02 * np.arange(200), index=idx, name="IBIT")
    iv = pd.Series(0.50, index=idx)
    regime = pd.Series(REGIME_NORMAL, index=idx)
    trades = run_backtest("TEST", spot, iv, regime, use_regime_filter=True)
    assert len(trades) > 0
    for t in trades:
        assert t.contracts >= 1
        assert t.short_strike > t.long_strike
        assert t.regime == REGIME_NORMAL


def test_run_backtest_regime_filter_blocks_crisis():
    idx = pd.bdate_range("2024-01-15", periods=120)
    spot = pd.Series(40 + 0.01 * np.arange(120), index=idx, name="IBIT")
    iv = pd.Series(0.55, index=idx)
    regime = pd.Series(REGIME_CRISIS, index=idx)
    trades = run_backtest("TEST", spot, iv, regime, use_regime_filter=True)
    assert len(trades) == 0  # all entries blocked by CRISIS sizing=0


def test_compute_metrics_handles_empty_trades():
    idx = pd.bdate_range("2024-01-15", periods=20)
    rets = pd.Series(0.0, index=idx)
    m = compute_metrics(rets, [])
    assert m["n_trades"] == 0
    assert m["sharpe"] == 0.0


def test_trades_to_daily_returns_attribution():
    idx = pd.bdate_range("2024-01-15", periods=20)
    spot = pd.Series(40 + 0.01 * np.arange(20), index=idx)
    iv = pd.Series(0.55, index=idx)
    regime = pd.Series(REGIME_NORMAL, index=idx)
    trades = run_backtest("TEST", spot, iv, regime, use_regime_filter=False)
    daily = trades_to_daily_returns(trades, idx)
    assert isinstance(daily, pd.Series)
    assert len(daily) == len(idx)
