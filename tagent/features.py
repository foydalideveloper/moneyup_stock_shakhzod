"""Feature engineering for the ML model.

Every feature uses only information available at or before each timestamp (uses
.shift()/.rolling() on past data), so there is no lookahead leakage. Input is an
OHLCV DataFrame (columns: open, high, low, close, volume); output is a feature
DataFrame aligned to the same index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)  # neutral when undefined


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def make_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        raise ValueError("df must have columns: open, high, low, close, volume")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    f = pd.DataFrame(index=df.index)

    # Returns over several horizons (current vs past — known at time t).
    f["ret_1"] = close.pct_change(1)
    f["ret_5"] = close.pct_change(5)
    f["ret_10"] = close.pct_change(10)
    f["log_ret_1"] = np.log(close / close.shift(1))

    # Volatility (rolling std of 1-bar returns).
    f["vol_10"] = f["ret_1"].rolling(10).std()
    f["vol_20"] = f["ret_1"].rolling(20).std()

    # Momentum.
    f["mom_5"] = close / close.shift(5) - 1.0
    f["mom_10"] = close / close.shift(10) - 1.0

    # Price vs moving averages.
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    f["close_sma10"] = close / sma_10 - 1.0
    f["close_sma20"] = close / sma_20 - 1.0

    # Indicators.
    f["rsi_14"] = _rsi(close, 14)
    macd, sig, hist = _macd(close)
    f["macd"] = macd
    f["macd_signal"] = sig
    f["macd_hist"] = hist

    # Range & position within recent range.
    f["hl_range"] = (high - low) / close
    roll_high = high.rolling(20).max()
    roll_low = low.rolling(20).min()
    rng = (roll_high - roll_low).replace(0.0, np.nan)
    f["pos_in_range"] = ((close - roll_low) / rng).fillna(0.5)

    # Volume relative to its recent average.
    vol_mean = volume.rolling(20).mean().replace(0.0, np.nan)
    f["vol_ratio"] = (volume / vol_mean).fillna(1.0)

    f = f.replace([np.inf, -np.inf], np.nan)
    if dropna:
        f = f.dropna()
    return f


def feature_columns(df_features: pd.DataFrame) -> list:
    return list(df_features.columns)
