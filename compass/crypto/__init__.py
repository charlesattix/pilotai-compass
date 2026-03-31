"""
compass.crypto — Crypto ETF composite scoring and risk management.

Data collectors (live API clients):
    coingecko      — BTC/ETH price, OHLC history, dominance (CoinGecko)
    fear_greed     — Crypto Fear & Greed Index (alternative.me)
    funding_rates  — Binance perp funding rates (BTC + ETH)
    deribit        — BTC options P/C ratio, max pain, OI by strike
    defi_llama     — Stablecoin supply (USDT+USDC+DAI) via DeFiLlama

Analysis modules:
    realized_vol   — Realized volatility and IV/RV spread
    regime         — Crypto regime classification
    composite_score — Weighted composite score engine (0-100)
    risk_gate      — Pre-entry risk gates for the crypto scanner
"""

from compass.crypto import coingecko, defi_llama, deribit, fear_greed, funding_rates  # noqa: F401
