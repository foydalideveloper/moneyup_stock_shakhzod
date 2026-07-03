"""Korean fundamental factors (free, via pykrx get_market_fundamental).

``get_market_fundamental(date, market)`` returns per-ticker BPS / PER / PBR / EPS /
DIV as of a date — enough to build the classic VALUE (cheap = low PBR/PER) and
QUALITY (profitable = high ROE ~ EPS/BPS) cross-sectional factors over our
point-in-time KR universe, survivorship-corrected via kr_universe membership.

Snapshots are reconstructed AS OF each monthly rebalance date and forward-filled to
the daily backtest index (a day uses the most recent snapshot <= it), so the factor
signals are strictly no-lookahead.

pykrx is imported lazily + injectable, so the package + tests run with no network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from tagent.config import DATA_DIR

FUND_CSV = "kr_fundamentals.csv"
METRICS = ["BPS", "PER", "PBR", "EPS", "DIV"]


def _import_pykrx():
    try:
        from pykrx import stock
    except ImportError as e:  # pragma: no cover
        raise ImportError("pykrx not installed. Run: pip install -r requirements.txt") from e
    return stock


def _yyyymmdd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _daystr(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _code(t) -> str:
    s = str(t)
    return s.zfill(6) if s.isdigit() else s


def fundamental_on(date, market: str = "KOSPI", stock=None) -> pd.DataFrame:
    """Per-ticker fundamentals as of ``date`` (BPS/PER/PBR/EPS/DIV), ticker-indexed."""
    stock = stock or _import_pykrx()
    df = stock.get_market_fundamental(_yyyymmdd(date), market=market)
    return df if df is not None else pd.DataFrame()


def build_fundamentals(dates: Sequence, markets: Sequence[str] = ("KOSPI",),
                       symbols: Optional[Sequence[str]] = None, stock=None) -> pd.DataFrame:
    """Long-format {date, ticker, BPS, PER, PBR, EPS, DIV} over the as-of ``dates``.
    Restricted to ``symbols`` when given. Degenerate/empty snapshots are skipped."""
    stock = stock or _import_pykrx()
    want = {_code(s) for s in symbols} if symbols else None
    rows: List[dict] = []
    for d in dates:
        for mk in markets:
            try:
                df = fundamental_on(d, mk, stock)
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue
            cols = {m: m for m in METRICS if m in df.columns}
            if not cols:
                continue
            for tk, r in df.iterrows():
                code = _code(tk)
                if want is not None and code not in want:
                    continue
                row = {"date": _daystr(d), "ticker": code}
                row.update({m: float(r.get(m)) if pd.notna(r.get(m)) else float("nan")
                            for m in cols})
                rows.append(row)
    return pd.DataFrame(rows, columns=["date", "ticker"] + METRICS)


def metric_panel(long_df: pd.DataFrame, metric: str, daily_index,
                 symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Wide [daily x symbol] panel of ``metric``, AS-OF forward-filled from the
    monthly snapshots onto ``daily_index`` (no-lookahead). NaN before first snapshot."""
    daily_index = pd.DatetimeIndex(daily_index)
    if long_df is None or long_df.empty or metric not in long_df.columns:
        cols = [_code(s) for s in symbols] if symbols else []
        return pd.DataFrame(index=daily_index, columns=cols, dtype=float)
    piv = long_df.pivot_table(index="date", columns="ticker", values=metric, aggfunc="last")
    piv.index = pd.to_datetime(piv.index)
    piv = piv.sort_index()
    if symbols is not None:
        piv = piv.reindex(columns=[_code(s) for s in symbols])
    full = piv.reindex(piv.index.union(daily_index)).sort_index().ffill()
    return full.reindex(daily_index).astype(float)


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def fundamentals_path(data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / FUND_CSV


def save_fundamentals(df: pd.DataFrame, data_dir=None) -> Path:
    path = fundamentals_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_fundamentals(data_dir=None) -> pd.DataFrame:
    path = fundamentals_path(data_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "ticker"] + METRICS)
    df = pd.read_csv(path, dtype={"ticker": str})
    df["ticker"] = df["ticker"].map(_code)
    return df
