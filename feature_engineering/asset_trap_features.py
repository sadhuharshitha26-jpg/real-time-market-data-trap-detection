"""
Asset-specific trap feature engineering.
Features adapt per cryptocurrency using the asset registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_registry import get_asset_params


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    return np.where(denominator == 0, default, numerator / denominator)


def compute_asset_trap_features(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Compute trap features tailored to the specific cryptocurrency.

    Args:
        df: DataFrame with OHLCV columns.
        symbol: Cryptocurrency symbol (e.g., "BTC-USD").
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    params = get_asset_params(symbol)
    features = df.copy()

    # Basic returns
    features["price_return"] = features["close"].pct_change()
    features["volume_change"] = features["volume"].pct_change()

    # 1) Fake breakout probability (asset-specific lookback & threshold)
    rolling_high = features["high"].rolling(window=params.breakout_lookback, min_periods=1).max()
    features["breakout_strength"] = _safe_divide(
        features["close"] - rolling_high, rolling_high
    )
    prev_high = rolling_high.shift(1)
    features["breakout_failure"] = (
        (features["high"] > prev_high) & (features["close"] < prev_high)
    ).astype(int)
    features["fake_breakout_prob"] = (
        features["breakout_failure"].rolling(window=params.reversal_window, min_periods=1).mean()
    )

    # 2) Volume absorption (asset-specific window)
    volume_mean = features["volume"].rolling(window=params.absorption_window, min_periods=1).mean()
    features["volume_spike"] = (features["volume"] > volume_mean * 1.5).astype(int)
    candle_range = (features["high"] - features["low"]).replace(0, np.nan)
    features["body_to_range"] = _safe_divide(
        (features["close"] - features["open"]).abs(), candle_range
    )
    features["absorption_signal"] = (
        (features["volume_spike"] == 1) & (features["body_to_range"] < 0.3)
    ).astype(int)
    features["volume_absorption_prob"] = (
        features["absorption_signal"].rolling(window=params.absorption_window, min_periods=1).mean()
    )

    # 3) Momentum exhaustion (asset-specific window & threshold)
    roc = features["close"].pct_change(periods=params.momentum_window) * 100
    features["momentum_roc"] = roc
    features["momentum_exhaustion"] = (
        (roc > params.momentum_threshold) & (roc.shift(1) > params.momentum_threshold) & (roc < roc.shift(1))
    ).astype(int)
    features["momentum_exhaustion_prob"] = (
        features["momentum_exhaustion"].rolling(window=params.momentum_window, min_periods=1).mean()
    )

    # 4) Reversal velocity (asset-specific window)
    features["reversal_velocity"] = (
        features["price_return"].rolling(window=params.reversal_window, min_periods=1).sum()
    )
    features["sharp_reversal"] = (
        features["reversal_velocity"] < -params.breakout_threshold / 100
    ).astype(int)

    # 5) Volatility compression → expansion
    volatility = features["high"].rolling(window=params.compression_window, min_periods=1).std() / features["close"]
    features["volatility"] = volatility
    features["compression_ratio"] = volatility.rolling(window=params.compression_window, min_periods=1).mean() / volatility
    features["compression_to_expansion"] = (
        features["compression_ratio"] > 1.5
    ).astype(int)

    # Clean up
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    return features
