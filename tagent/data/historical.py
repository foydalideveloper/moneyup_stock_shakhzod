"""Historical OHLCV loader (offline training/backtesting layer).

Uses yfinance (free) by default. Returns a dict {symbol: DataFrame} where each
DataFrame has columns: open, high, low, close, volume — indexed by timestamp.
Saved as CSV so no extra parquet dependency is needed.

yfinance is fine HERE (offline history). It is NOT used for the live feed —
that's the Alpaca/Kiwoom streaming adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from tagent.config import DATA_DIR

_COLS = ["open", "high", "low", "close", "volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance sometimes returns multi-index columns even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    keep = [c for c in _COLS if c in df.columns]
    df = df[keep].copy()
    df = df.dropna()
    df.index.name = "timestamp"
    return df


def download_history(symbols: List[str], start: Optional[str] = None,
                     end: Optional[str] = None, period: str = "2y",
                     interval: str = "1d", save: bool = True) -> Dict[str, pd.DataFrame]:
    """Download OHLCV history. If `start` is given, `period` is ignored."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("yfinance not installed. Run: pip install -r requirements.txt") from e

    symbols = [s.strip().upper() for s in symbols if s.strip()]
    raw = yf.download(
        tickers=symbols, start=start, end=end,
        period=(None if start else period), interval=interval,
        group_by="ticker", auto_adjust=True, progress=False, threads=True,
    )

    out: Dict[str, pd.DataFrame] = {}
    if len(symbols) == 1:
        out[symbols[0]] = _clean(raw.copy())
    else:
        top = set(raw.columns.get_level_values(0))
        for sym in symbols:
            if sym in top:
                out[sym] = _clean(raw[sym].copy())

    if save:
        for sym, df in out.items():
            df.to_csv(DATA_DIR / f"{sym}_{interval}.csv")
    return out


def load_history(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Load previously downloaded CSV history for a symbol."""
    path = Path(DATA_DIR) / f"{symbol.upper()}_{interval}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No saved data at {path}. Run download_data.py first.")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return _clean(df)
