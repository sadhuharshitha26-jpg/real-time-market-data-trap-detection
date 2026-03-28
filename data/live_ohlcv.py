"""
Live OHLCV fetcher for supported cryptocurrencies.
Uses public Binance API; fallback to simulated data if API fails.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import requests

from asset_registry import supported_symbols


def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
    """Fetch recent OHLCV from Binance."""
    # Convert MarketTrap symbol to Binance format (e.g., BTC-USD -> BTCUSDT)
    binance_symbol = symbol.replace("-", "")
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"Failed to fetch {symbol} from Binance: {e}")
        return None


def simulate_ohlcv(symbol: str, periods: int = 100) -> pd.DataFrame:
    """Generate realistic simulated OHLCV for fallback/demo."""
    import numpy as np
    np.random.seed(hash(symbol) % 2**32)
    base_price = {
        "BTC-USD": 43000,
        "ETH-USD": 2600,
        "SOL-USD": 105,
        "BNB-USD": 310,
        "XRP-USD": 0.62,
    }.get(symbol, 100)
    timestamps = pd.date_range(end=datetime.utcnow(), periods=periods, freq="1min")
    price = np.cumprod(1 + np.random.randn(periods) * 0.001) * base_price
    volume = np.random.randint(1_000_000, 10_000_000, size=periods)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": price,
        "high": price * (1 + np.random.rand(periods) * 0.002),
        "low": price * (1 - np.random.rand(periods) * 0.002),
        "close": price,
        "volume": volume,
    })
    return df


def get_live_ohlcv(symbol: str, periods: int = 100) -> pd.DataFrame:
    """Get live OHLCV: try Binance first, fallback to simulation."""
    df = fetch_binance_klines(symbol, limit=periods)
    if df is None or df.empty:
        print(f"Falling back to simulated data for {symbol}")
        df = simulate_ohlcv(symbol, periods)
    return df


def stream_live_ohlcv(symbol: str, interval_seconds: int = 5, max_batches: int = None):
    """Generator yielding live OHLCV updates for a symbol."""
    batch_count = 0
    while True:
        df = get_live_ohlcv(symbol, periods=100)
        yield df
        batch_count += 1
        if max_batches and batch_count >= max_batches:
            break
        time.sleep(interval_seconds)
