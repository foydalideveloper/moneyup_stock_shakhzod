"""Classic 12-1 cross-sectional momentum on STOCKS (the documented equity factor).

Crypto momentum failed walk-forward validation (short history). Stocks are where
cross-sectional momentum is the textbook factor and we have ~10 years of daily
data (Korean 50-name universe from 2016 + the US watchlist). This wires the
existing engine (tagent.xs_momentum) + validator (tagent.xs_momentum_validate) to
the stock panels with the classic setting:

  * formation = ~12 months (252d) SKIPPING the most recent ~1 month (21d) — "12-1",
  * MONTHLY rebalance (~21 trading days), low turnover,
  * LONG top quantile / SHORT bottom (research), plus a LONG-ONLY top-quantile
    variant (the deployable version in Korea, where shorting is restricted),
  * realistic stock costs (commission + spread; KR also has a sell tax).

Pure pandas (loaders + a config helper); unit-tested on synthetic panels.

Survivorship caveat: the fixed 50-name KR / 10-name US universes are TODAY's liquid
names, so the backtest never holds since-delisted losers — a known upward bias in
any fixed-universe study. Treat absolute returns as optimistic.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from tagent.config import DATA_DIR
from tagent.xs_momentum import XSMomConfig

STOCK_PPY = 252                          # trading days / year

# Realistic per-turnover costs. KR: ~1.5bp commission + ~18bp sell tax (amortised)
# + spread; US: ~free commission + spread/slippage.
KR_COST = {"cost_bps": 15.0, "slippage_bps": 5.0}
US_COST = {"cost_bps": 2.0, "slippage_bps": 3.0}

_NON_STOCK_PREFIXES = ("klines_", "funding_", "microstructure_")


def classic_config(market: str = "kr", lookback: int = 252, allow_short: bool = False,
                   top_q: float = 0.2, hysteresis: float = 0.1) -> XSMomConfig:
    """Classic 12-1 equity momentum config: 252d formation, skip 21d, monthly
    rebalance, stock costs. ``allow_short=False`` is the deployable KR variant."""
    cost = KR_COST if str(market).lower() == "kr" else US_COST
    return XSMomConfig(lookback=lookback, skip_recent=21, rebalance=21, top_q=top_q,
                       hysteresis=hysteresis, allow_short=allow_short, **cost)


def _is_kr(name: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", name))


def load_stock_panel(market: str = "kr", data_dir=None, symbols: Optional[List[str]] = None,
                     min_bars: int = 300, fields: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load daily OHLCV CSVs (``<sym>_1d.csv``) into {symbol: df}.

    ``fields`` selects which columns to keep (default ['close'] for momentum/factors;
    pass e.g. ['open','close'] for the overnight / lead-lag daily strategies).

    market='kr' takes the 6-digit KRX codes; 'us' takes alpha tickers (excluding
    SPY, used as a benchmark, and any crypto/funding/microstructure files).
    """
    fields = fields or ["close"]
    base = data_dir if data_dir is not None else DATA_DIR
    out: Dict[str, pd.DataFrame] = {}
    if symbols:
        files = [(s, os.path.join(base, f"{s}_1d.csv")) for s in symbols]
    else:
        files = []
        for p in sorted(glob.glob(os.path.join(base, "*_1d.csv"))):
            name = os.path.basename(p)[:-len("_1d.csv")]
            if any(name.startswith(pre) for pre in _NON_STOCK_PREFIXES):
                continue
            if str(market).lower() == "kr" and _is_kr(name):
                files.append((name, p))
            elif str(market).lower() == "us" and not _is_kr(name) and name.upper() != "SPY":
                files.append((name, p))
    for sym, p in files:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, index_col="timestamp", parse_dates=True).sort_index()
        except Exception:
            continue
        keep = [f for f in fields if f in df.columns]
        if "close" in keep and len(df) >= min_bars:
            out[sym] = df[keep].astype(float)
    return out


def load_benchmark(symbol: str = "SPY", data_dir=None) -> Optional[pd.Series]:
    """Daily returns of a real index benchmark (e.g. SPY), or None if not cached."""
    base = data_dir if data_dir is not None else DATA_DIR
    path = os.path.join(base, f"{symbol}_1d.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True).sort_index()
        return df["close"].astype(float).pct_change().dropna()
    except Exception:
        return None
