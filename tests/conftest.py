"""Shared test fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_config():
    return {
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'strategy': {
            'min_dte': 30,
            'max_dte': 45,
            'manage_dte': 21,
            'min_delta': 0.10,
            'max_delta': 0.15,
            'spread_width': 5,
            'min_iv_rank': 25,
            'min_iv_percentile': 25,
            'technical': {
                'use_trend_filter': True,
                'use_rsi_filter': True,
                'use_support_resistance': True,
                'fast_ma': 20,
                'slow_ma': 50,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            },
        },
        'risk': {
            'account_size': 100000,
            'max_risk_per_trade': 2.0,
            'max_positions': 7,
            'profit_target': 50,
            'stop_loss_multiplier': 2.5,
            'delta_threshold': 0.30,
            'min_credit_pct': 20,
        },
        'alerts': {
            'output_json': True, 'output_text': True, 'output_csv': True,
            'json_file': 'alerts.json', 'text_file': 'alerts.txt', 'csv_file': 'alerts.csv',
            'telegram': {'enabled': False, 'bot_token': '', 'chat_id': ''},
        },
        'alpaca': {'enabled': False, 'api_key': '', 'api_secret': '', 'paper': True},
        'data': {
            'provider': 'yfinance',
            'backtest_lookback': 365,
            'use_cache': True,
            'cache_expiry_minutes': 15,
        },
        'logging': {'level': 'WARNING', 'file': '/tmp/test_trading.log', 'console': False},
        'backtest': {
            'starting_capital': 100000,
            'commission_per_contract': 0.65,
            'slippage': 0.05,
            'generate_reports': False,
            'report_dir': '/tmp/backtest_reports',
        },
    }

@pytest.fixture
def sample_price_data():
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='B')
    close = 450 + np.cumsum(np.random.randn(100) * 2)
    return pd.DataFrame({
        'Open': close - np.random.rand(100),
        'High': close + np.abs(np.random.randn(100)),
        'Low': close - np.abs(np.random.randn(100)),
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 100),
    }, index=dates)
